/**
 *  @file include/numkong/attention/serial.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Serial (SIMD-free) ragged attention baseline.
 *
 *  @sa include/numkong/attention.h
 *
 *  Width-agnostic reference implementation of the ragged scaled-dot-product attention family: any
 *  `depth ≥ 1`, any segment lengths, GQA/MQA, the same base-2 softmax formulation and the same
 *  [tasks_begin, tasks_end) windows over the flat @b [query_tokens,heads] grid as the SIMD
 *  backends, under any band of visible keys.
 *
 *  @section attention_serial_roles Roles in the Family
 *
 *  1. Ground truth for every SIMD backend's conformance tests.
 *  2. Final runtime-dispatch fallback on CPUs without any compiled SIMD target.
 *  3. Fallback for shapes outside a SIMD backend's fast-path envelope (`depth > 256`), invoked from
 *     those backends so pack and attention always agree on the packed-buffer format.
 *
 *  @section attention_serial_layout Packed Layout
 *
 *  This file also owns the family-shared packed-KV header and offsets layout:
 *
 *  @verbatim
 *  [64 B header]
 *  [directory: u64 payload_offsets[S] + u32 key_offsets[S+1] + u32 key_lengths[S], 64-byte padded]
 *  [per-backend payload]
 *  @endverbatim
 *
 *  The payload is backend-opaque; serial stores K and V as plain F32 row-major planes
 *  `[key_value_head][position][channel]` per segment, so both input dtypes — BF16, E4M3 — share one
 *  compute path after per-element conversion at pack.
 *
 *  @section attention_serial_scaled Block-Scaled Planes
 *
 *  NVFP4 planes hold each element times its UE4M3 block scale, which F32 keeps exactly, and the
 *  header keeps both tensor scales. MX planes hold each element times its UE8M0 block scale rebased
 *  by the plane's largest finite block exponent less 31, kept in an `i8 plane_exponents[S][2][H]`
 *  table, 64-byte padded, past the bound of the planes. A plane whose block exponents span more
 *  than 89 binades keeps −128 there instead, and each of its rows starts with the row's raw codes,
 *  then its scale codes. Scores sum each query block in F32 and add the blocks into a wide sum, so
 *  query scales of any spread round once.
 *
 *  Like all serial kernels, it avoids libm: the base-2 exponent uses the same degree-4 polynomial
 *  and the same denormal-avoiding clamp as the AVX-512 helper, so serial and vector paths agree to
 *  polynomial precision.
 *
 *  @sa nk_f32_exp2_serial_
 *
 *  @section attention_serial_i8 I8 Weight Quantization
 *
 *  The I8 path keeps scores exact in I32 integer arithmetic and quantizes softmax weights to U8 as
 *  round(255 · 2^(s₂ − m₂)); the max-scoring position always lands on 255, so the weight sum can
 *  never be zero. Normalizing by that sum cancels the 255, so no descale constant remains.
 */
#ifndef NUMKONG_ATTENTION_SERIAL_H
#define NUMKONG_ATTENTION_SERIAL_H

#include "numkong/types.h"
#include "numkong/capabilities.h"       // `nk_capability_t`
#include "numkong/scalar/serial.h"      // `nk_f32_exp2_serial_`, `NUMKONG_F32_LOG2E_`
#include "numkong/cast/serial.h"        // `nk_bf16_to_f32_serial_`, `nk_e4m3_to_f32_serial_`
#include "numkong/probability/serial.h" // `nk_f32_log_serial_`
#include "numkong/dots/serial.h"        // `nk_cross_wide_add_serial_`, `nk_cross_scaled_exact_wide_mxfp4_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/** Packed ragged KV cache header (64 bytes), shared by all attention backends. Followed by the
 *  segment directory; the payload beyond it is backend-specific. */
typedef struct {

    /** Number of K/V heads, ≤ query heads for GQA. */
    nk_u32_t key_value_head_count;

    /** Head dimension the buffer was packed for. */
    nk_u32_t depth;

    /** Number of independent segments packed. */
    nk_u32_t segments;

    /** The tensor scales of NVFP4 keys and values, zero for every other dtype. */
    nk_f32_t key_tensor_scale, value_tensor_scale;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[9];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_attention_packed_header_t;

nk_static_assert_(sizeof(nk_attention_packed_header_t) == 64, nk_attention_packed_header_t_must_be_64_bytes);

/** Directory size in bytes, 64-byte padded: @p segment_count u64 payload offsets, @p segment_count
 *  + 1 u32 key offsets, then @p segment_count u32 key lengths. */
NUMKONG_INLINE nk_size_t nk_attention_pack_directory_size_serial_(nk_size_t segment_count) NUMKONG_STREAMABLE_ {
    return nk_size_round_up_to_multiple_(segment_count * sizeof(nk_u64_t) + (2 * segment_count + 1) * sizeof(nk_u32_t),
                                         64);
}

/** The pack's payload offsets, one per segment, in bytes from the start of the payload. */
NUMKONG_INLINE nk_u64_t const *nk_attention_packed_payload_offsets_serial_(void const *key_value_packed)
    NUMKONG_STREAMABLE_ {
    return (nk_u64_t const *)((char const *)key_value_packed + sizeof(nk_attention_packed_header_t));
}

/** The pack's @p segment_count + 1 key slot boundaries. */
NUMKONG_INLINE nk_u32_t const *nk_attention_packed_key_offsets_serial_(void const *key_value_packed,
                                                                       nk_size_t segment_count) NUMKONG_STREAMABLE_ {
    return (nk_u32_t const *)(nk_attention_packed_payload_offsets_serial_(key_value_packed) + segment_count);
}

/** The pack's @p segment_count key counts, one per segment. */
NUMKONG_INLINE nk_u32_t const *nk_attention_packed_key_lengths_serial_(void const *key_value_packed,
                                                                       nk_size_t segment_count) NUMKONG_STREAMABLE_ {
    return nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count) + segment_count + 1;
}

/** Keys of segment @p segment_idx at pack time: its @p key_lengths entry, or its slot if null. */
NUMKONG_INLINE nk_size_t nk_attention_pack_key_count_serial_(nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                             nk_size_t segment_idx) NUMKONG_STREAMABLE_ {
    return key_lengths ? key_lengths[segment_idx] : key_offsets[segment_idx + 1] - key_offsets[segment_idx];
}

/** Whether the counts fit the header's u32 fields and the slots are ordered and hold their keys. */
NUMKONG_INLINE nk_status_t nk_attention_pack_validate_serial_(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                              nk_size_t segment_count) {
    if (key_value_head_count > NUMKONG_U32_MAX || depth > NUMKONG_U32_MAX || segment_count > NUMKONG_U32_MAX)
        return nk_unexpected_dimensions_k;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const slot_end = key_offsets[segment_idx + 1];
        if (key_offsets[segment_idx] > slot_end) return nk_unexpected_dimensions_k;
        if (key_lengths && (nk_size_t)key_offsets[segment_idx] + key_lengths[segment_idx] > slot_end)
            return nk_unexpected_dimensions_k;
    }
    return nk_success_k;
}

/** Payload bytes of a segment of @p length keys: the K and V planes of every head, each padded to
 *  @p position_multiple positions of @p unit_bytes, the formula every backend's directory sums. */
NUMKONG_CONSTEXPR nk_u64_t nk_attention_pack_segment_bytes_serial_(nk_size_t length, nk_size_t key_value_head_count,
                                                                   nk_size_t position_multiple,
                                                                   nk_size_t unit_bytes) NUMKONG_STREAMABLE_ {
    return 2 * key_value_head_count * (nk_u64_t)nk_size_round_up_to_multiple_(length, position_multiple) * unit_bytes;
}

/** Bytes enough for a pack of @p segment_count segments holding @p token_count keys however they
 *  split: header, directory, and the K and V planes of every head, each segment padded to
 *  @p position_multiple positions of @p unit_bytes, which pads it by at most one less. */
NUMKONG_INLINE nk_size_t nk_attention_pack_bound_serial_(nk_size_t key_value_head_count, nk_size_t token_count,
                                                         nk_size_t segment_count, nk_size_t position_multiple,
                                                         nk_size_t unit_bytes) NUMKONG_STREAMABLE_ {
    nk_size_t const positions = token_count + segment_count * (position_multiple - 1);
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_serial_(segment_count) +
           2 * key_value_head_count * positions * unit_bytes;
}

/**
 *  @brief Writes the family-shared packed-KV header and directory, recording the packing
 *      @p capability, when the window starts at task 0.
 *
 *  The one place a null @p key_lengths resolves to the slot widths. No other window reads the
 *  directory: each sums @c nk_attention_pack_segment_bytes_serial_ over the segments before its
 *  own, so windows run in any order, like @c nk_dots_pack ones.
 */
NUMKONG_INLINE void nk_attention_pack_directory_serial_(void *key_value_packed, nk_size_t key_value_head_count,
                                                        nk_size_t depth, nk_u32_t const *key_offsets,
                                                        nk_u32_t const *key_lengths, nk_size_t segment_count,
                                                        nk_size_t tasks_begin, nk_size_t position_multiple,
                                                        nk_size_t unit_bytes, nk_capability_t capability) {
    if (tasks_begin != 0) return;
    // Zero the whole directory, header plus the table including its 64-byte-aligned tail, so the
    // packed blob is a pure function of its inputs, with no allocator garbage in the unwritten
    // slack. The per-segment payload planes are zero-filled by each backend, so this makes the pack
    // hermetic and the caller need not pre-zero the buffer.

    nk_size_t const directory_bytes = sizeof(nk_attention_packed_header_t) +
                                      nk_attention_pack_directory_size_serial_(segment_count);
    char *packed_bytes = (char *)key_value_packed;
    for (nk_size_t byte_index = 0; byte_index < directory_bytes; byte_index++) packed_bytes[byte_index] = 0;
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    header->key_value_head_count = (nk_u32_t)key_value_head_count;
    header->depth = (nk_u32_t)depth;
    header->segments = (nk_u32_t)segment_count;
    header->capability = capability;
    nk_u64_t *payload_offsets = (nk_u64_t *)((char *)key_value_packed + sizeof(*header));
    nk_u32_t *offsets_copy = (nk_u32_t *)(payload_offsets + segment_count);
    nk_u32_t *lengths_copy = offsets_copy + segment_count + 1;
    nk_u64_t running = 0;
    nk_size_t token_count = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const key_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        token_count += key_count;
        payload_offsets[segment_idx] = running;
        offsets_copy[segment_idx] = key_offsets[segment_idx];
        lengths_copy[segment_idx] = (nk_u32_t)key_count;
        running += nk_attention_pack_segment_bytes_serial_(key_count, key_value_head_count, position_multiple,
                                                           unit_bytes);
    }
    offsets_copy[segment_count] = key_offsets[segment_count];
    // The slack past the used payload, up to the size bound, is zeroed too.
    nk_size_t const bound_bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count,
                                                                  position_multiple, unit_bytes);
    for (nk_size_t byte_index = directory_bytes + running; byte_index < bound_bytes; byte_index++)
        packed_bytes[byte_index] = 0;
}

/**
 *  @brief Reads a packed KV cache's shape from its header.
 *
 *  Shared by every per-(dtype, ISA) nk_attention_packed_shape_serial_* accessor.
 */
NUMKONG_INLINE void nk_attention_packed_shape_serial_(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                      nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    *key_value_head_count = header->key_value_head_count;
    *depth = header->depth;
    *segments = header->segments;
}

/** Whether @p key_value_packed was packed by @p capability, the layout its consumer reads. */
NUMKONG_INLINE int nk_attention_packed_by_serial_(void const *key_value_packed, nk_capability_t capability) {
    return ((nk_attention_packed_header_t const *)key_value_packed)->capability == capability;
}

/** The query rows @p query_offsets address across all segments of @p key_value_packed. */
NUMKONG_INLINE nk_size_t nk_attention_query_end_serial_(void const *key_value_packed, nk_u32_t const *query_offsets) {
    nk_size_t const segment_count = ((nk_attention_packed_header_t const *)key_value_packed)->segments;
    return query_offsets[segment_count];
}

/** The position of a segment's first query row: queries align to the end of the @p key_count keys,
 *  so the first of @p query_count rows sits at @p key_count − @p query_count, which lies before
 *  key 0 when that is negative. */
NUMKONG_INLINE nk_i64_t nk_attention_first_position_serial_(nk_size_t query_count,
                                                            nk_size_t key_count) NUMKONG_STREAMABLE_ {
    return (nk_i64_t)key_count - (nk_i64_t)query_count;
}

/** The segment holding query token @p token: the last of @p segment_count whose first row in
 *  @p query_offsets is at most @p token, so empty segments before it are skipped. */
NUMKONG_CONSTEXPR nk_size_t nk_attention_segment_of_serial_(nk_u32_t const *query_offsets, nk_size_t segment_count,
                                                            nk_size_t token) NUMKONG_STREAMABLE_ {
    nk_size_t low = 0, high = segment_count;
    while (high - low > 1) {
        nk_size_t const middle = low + (high - low) / 2;
        if (query_offsets[middle] <= token) low = middle;
        else high = middle;
    }
    return low;
}

/** The natural log-sum-exp of a row whose base-2 scores peak at @p max2, given @p sum of their
 *  2^(score₂ − @p max2) weights: −∞ for a row that saw no key. */
NUMKONG_INLINE nk_f32_t nk_attention_log_sum_exp_serial_(nk_f32_t max2, nk_f32_t sum) NUMKONG_STREAMABLE_ {
    return sum > 0 ? max2 * NUMKONG_F32_LN2_ + nk_f32_log_serial_(sum) : -NUMKONG_F32_INF;
}

NUMKONG_INLINE nk_f32_t nk_attention_load_bf16_serial_(void const *element) {
    nk_f32_t result;
    nk_bf16_to_f32_serial_((nk_bf16_t const *)element, &result);
    return result;
}

NUMKONG_INLINE nk_f32_t nk_attention_load_f16_serial_(void const *element) {
    nk_f32_t result;
    nk_f16_to_f32_serial_((nk_f16_t const *)element, &result);
    return result;
}

NUMKONG_INLINE nk_f32_t nk_attention_load_e4m3_serial_(void const *element) {
    nk_f32_t result;
    nk_e4m3_to_f32_serial_((nk_e4m3_t const *)element, &result);
    return result;
}

enum {

    /** Widest span of block exponents an MX plane rebases across into F32: rebased E5M2 elements
     *  then lie within 2⁻⁷⁴ … 2⁴⁷, so their products with any query code stay normal and exact. */
    nk_attention_plane_window_k_ = 89,

    /** The plane-exponent entry of an MX plane kept as raw codes and scale codes. */
    nk_attention_raw_plane_k_ = -128,
};

/** Bytes of an MX pack's plane-exponent table, 64-byte padded: one entry per K plane and per V
 *  plane, each segment's K planes first. */
NUMKONG_INLINE nk_size_t nk_attention_plane_exponents_size_serial_(nk_size_t key_value_head_count,
                                                                   nk_size_t segment_count) NUMKONG_STREAMABLE_ {
    return nk_size_round_up_to_multiple_(2 * key_value_head_count * segment_count, 64);
}

/** Where an MX pack's plane-exponent table starts: at the bound of its planes, padded to
 *  @p position_multiple positions of @p unit_bytes, over the keys of @p key_offsets and
 *  @p key_lengths, past the slack the directory zeroes. */
NUMKONG_INLINE nk_size_t nk_attention_plane_exponents_offset_serial_(nk_size_t key_value_head_count,
                                                                     nk_size_t position_multiple, nk_size_t unit_bytes,
                                                                     nk_u32_t const *key_offsets,
                                                                     nk_u32_t const *key_lengths,
                                                                     nk_size_t segment_count) NUMKONG_STREAMABLE_ {
    nk_size_t token_count = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        token_count += nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
    return nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, position_multiple,
                                           unit_bytes);
}

/** The plane-exponent entry of @p rows rows of @p blocks UE8M0 scale codes, @p stride bytes apart:
 *  their largest finite exponent, 0 when none is, or @c nk_attention_raw_plane_k_ when they span
 *  more than @c nk_attention_plane_window_k_ binades. */
NUMKONG_INLINE nk_i8_t nk_attention_plane_exponent_ue8m0_serial_(nk_u8_t const *scales, nk_size_t stride,
                                                                 nk_size_t rows, nk_size_t blocks) NUMKONG_STREAMABLE_ {
    nk_i32_t low = 255, high = 0;
    for (nk_size_t row = 0; row < rows; row++)
        for (nk_size_t block = 0; block < blocks; block++) {
            nk_i32_t const code = scales[row * stride + block];
            if (code == 0 || code == 255) continue;
            low = code < low ? code : low, high = code > high ? code : high;
        }
    if (low > high) return 0;
    return (nk_i8_t)(high - low > nk_attention_plane_window_k_ ? nk_attention_raw_plane_k_ : high - 127);
}

/** The power of two an MX plane's F32 values sit under: its entry less 31, or 0 for raw planes. */
NUMKONG_INLINE nk_i32_t nk_attention_plane_base_serial_(nk_i8_t exponent) NUMKONG_STREAMABLE_ {
    return exponent == nk_attention_raw_plane_k_ ? 0 : exponent - nk_cross_scaled_headroom_k;
}

/** Fills a @p row_bytes long row of a raw plane with @p codes_bytes of @p codes, then @p blocks
 *  scale codes, then zeros. */
NUMKONG_INLINE void nk_attention_raw_row_serial_(nk_u8_t const *codes, nk_size_t codes_bytes, nk_u8_t const *scales,
                                                 nk_size_t blocks, nk_size_t row_bytes, void *row) {
    nk_u8_t *bytes = (nk_u8_t *)row;
    for (nk_size_t byte = 0; byte < row_bytes; byte++)
        bytes[byte] = byte < codes_bytes ? codes[byte] : byte < codes_bytes + blocks ? scales[byte - codes_bytes] : 0;
}

/** Element @p index of an NVFP4 row of @p codes times its UE4M3 block scale, exactly. */
NUMKONG_INLINE nk_f32_t nk_attention_load_nvfp4_serial_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t index) {
    nk_i32_t exponent;
    return nk_e2m1_load_f32_serial_(codes, index) * nk_ue4m3_split_serial_(scales[index / 16], &exponent);
}

/** Element @p index of an MXFP4 row of @p codes times its block scale, rebased by @p base. */
NUMKONG_INLINE nk_f32_t nk_attention_load_mxfp4_serial_(nk_u8_t const *codes, nk_u8_t const *scales, nk_size_t index,
                                                        nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_serial_(scales[index / 32], &exponent);
    return nk_scale_f32_serial_(nk_e2m1_load_f32_serial_(codes, index) * mantissa, exponent - base);
}

/** Element @p index of a packed MXFP4 row of @p depth elements in a plane at @p exponent, under the
 *  plane's base. */
NUMKONG_INLINE nk_f32_t nk_attention_plane_mxfp4_serial_(nk_u8_t const *row, nk_i8_t exponent, nk_size_t depth,
                                                         nk_size_t index) {
    return exponent == nk_attention_raw_plane_k_ ? nk_attention_load_mxfp4_serial_(row, row + depth / 2, index, 0)
                                                 : ((nk_f32_t const *)row)[index];
}

/** The base-2 score of an MXFP4 query row against a packed key row in a plane at @p exponent:
 *  F32 sums per query block, each added at its exponent plus the plane's base into a wide sum, or
 *  the exact dot of raw codes on both sides for a raw plane, rounded once times @p scale2. */
NUMKONG_INLINE nk_f32_t nk_attention_score_mxfp4_serial_(nk_u8_t const *query_row, nk_u8_t const *query_scales,
                                                         nk_u8_t const *key_row, nk_i8_t exponent, nk_size_t depth,
                                                         nk_f32_t scale2) {
    nk_cross_wide_sum_t dot = {0, 0};
    if (exponent == nk_attention_raw_plane_k_) {
        nk_cross_wide_sum_t query_sumsq, key_sumsq;
        dot = nk_cross_scaled_exact_wide_mxfp4_serial_(query_row, query_scales, key_row, key_row + depth / 2, depth,
                                                       &query_sumsq, &key_sumsq);
    }
    else
        for (nk_size_t block = 0; block * 32 < depth; block++) {
            nk_i32_t query_exponent;
            nk_f32_t const mantissa = nk_ue8m0_split_serial_(query_scales[block], &query_exponent);
            nk_f32_t block_sum = 0;
            for (nk_size_t index = block * 32; index != (block + 1) * 32; index++)
                block_sum += nk_e2m1_load_f32_serial_(query_row, index) * ((nk_f32_t const *)key_row)[index];
            nk_cross_wide_add_serial_(&dot, block_sum * mantissa,
                                      query_exponent + exponent - nk_cross_scaled_headroom_k);
        }
    return nk_scale_f32_serial_(dot.sum * scale2, dot.exponent);
}

/** Packs @p positions MXFP4 rows of head @p head_idx from row @p first_position of @p operand into
 *  @p plane and returns its exponent entry: elements times their block scales under the plane's
 *  base, or each row's raw codes and scale codes when the entry is raw. */
NUMKONG_INLINE nk_i8_t nk_attention_pack_plane_mxfp4_serial_(nk_mxfp4_cref_t const *operand, nk_size_t stride,
                                                             nk_size_t first_position, nk_size_t positions,
                                                             nk_size_t head_idx, nk_size_t depth, nk_f32_t *plane) {
    nk_size_t const scales_stride = stride / 16, blocks = depth / 32, codes_bytes = depth / 2;
    nk_u8_t const *codes = (nk_u8_t const *)operand->elements + first_position * stride + head_idx * codes_bytes;
    nk_u8_t const *scales = (nk_u8_t const *)operand->scales + first_position * scales_stride + head_idx * blocks;
    nk_i8_t const exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, positions, blocks);
    nk_i32_t const base = nk_attention_plane_base_serial_(exponent);
    for (nk_size_t position_idx = 0; position_idx < positions; position_idx++) {
        nk_u8_t const *row_codes = codes + position_idx * stride, *row_scales = scales + position_idx * scales_stride;
        nk_f32_t *row = plane + position_idx * depth;
        if (exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_row_serial_(row_codes, codes_bytes, row_scales, blocks, depth * sizeof(nk_f32_t), row);
        else
            for (nk_size_t index = 0; index < depth; index++)
                row[index] = nk_attention_load_mxfp4_serial_(row_codes, row_scales, index, base);
    }
    return exponent;
}

/** Element @p index of an MXFP6 E2M3 row of @p codes times its block scale, rebased by @p base. */
NUMKONG_INLINE nk_f32_t nk_attention_load_mxfp6e2m3_serial_(nk_u8_t const *codes, nk_u8_t const *scales,
                                                            nk_size_t index, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_serial_(scales[index / 32], &exponent);
    return nk_scale_f32_serial_(nk_e2m3_load_f32_serial_(codes, index) * mantissa, exponent - base);
}

/** @c nk_attention_plane_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE nk_f32_t nk_attention_plane_mxfp6e2m3_serial_(nk_u8_t const *row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_size_t index) {
    return exponent == nk_attention_raw_plane_k_ ? nk_attention_load_mxfp6e2m3_serial_(row, row + depth, index, 0)
                                                 : ((nk_f32_t const *)row)[index];
}

/** @c nk_attention_score_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE nk_f32_t nk_attention_score_mxfp6e2m3_serial_(nk_u8_t const *query_row, nk_u8_t const *query_scales,
                                                             nk_u8_t const *key_row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_f32_t scale2) {
    nk_cross_wide_sum_t dot = {0, 0};
    if (exponent == nk_attention_raw_plane_k_) {
        nk_cross_wide_sum_t query_sumsq, key_sumsq;
        dot = nk_cross_scaled_exact_wide_mxfp6e2m3_serial_(query_row, query_scales, key_row, key_row + depth, depth,
                                                           &query_sumsq, &key_sumsq);
    }
    else
        for (nk_size_t block = 0; block * 32 < depth; block++) {
            nk_i32_t query_exponent;
            nk_f32_t const mantissa = nk_ue8m0_split_serial_(query_scales[block], &query_exponent);
            nk_f32_t block_sum = 0;
            for (nk_size_t index = block * 32; index != (block + 1) * 32; index++)
                block_sum += nk_e2m3_load_f32_serial_(query_row, index) * ((nk_f32_t const *)key_row)[index];
            nk_cross_wide_add_serial_(&dot, block_sum * mantissa,
                                      query_exponent + exponent - nk_cross_scaled_headroom_k);
        }
    return nk_scale_f32_serial_(dot.sum * scale2, dot.exponent);
}

/** @c nk_attention_pack_plane_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE nk_i8_t nk_attention_pack_plane_mxfp6e2m3_serial_(nk_mxfp6e2m3_cref_t const *operand, nk_size_t stride,
                                                                 nk_size_t first_position, nk_size_t positions,
                                                                 nk_size_t head_idx, nk_size_t depth, nk_f32_t *plane) {
    nk_size_t const scales_stride = stride / 32, blocks = depth / 32, codes_bytes = depth;
    nk_u8_t const *codes = (nk_u8_t const *)operand->elements + first_position * stride + head_idx * codes_bytes;
    nk_u8_t const *scales = (nk_u8_t const *)operand->scales + first_position * scales_stride + head_idx * blocks;
    nk_i8_t const exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, positions, blocks);
    nk_i32_t const base = nk_attention_plane_base_serial_(exponent);
    for (nk_size_t position_idx = 0; position_idx < positions; position_idx++) {
        nk_u8_t const *row_codes = codes + position_idx * stride, *row_scales = scales + position_idx * scales_stride;
        nk_f32_t *row = plane + position_idx * depth;
        if (exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_row_serial_(row_codes, codes_bytes, row_scales, blocks, depth * sizeof(nk_f32_t), row);
        else
            for (nk_size_t index = 0; index < depth; index++)
                row[index] = nk_attention_load_mxfp6e2m3_serial_(row_codes, row_scales, index, base);
    }
    return exponent;
}

/** Element @p index of an MXFP6 E3M2 row of @p codes times its block scale, rebased by @p base. */
NUMKONG_INLINE nk_f32_t nk_attention_load_mxfp6e3m2_serial_(nk_u8_t const *codes, nk_u8_t const *scales,
                                                            nk_size_t index, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_serial_(scales[index / 32], &exponent);
    return nk_scale_f32_serial_(nk_e3m2_load_f32_serial_(codes, index) * mantissa, exponent - base);
}

/** @c nk_attention_plane_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE nk_f32_t nk_attention_plane_mxfp6e3m2_serial_(nk_u8_t const *row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_size_t index) {
    return exponent == nk_attention_raw_plane_k_ ? nk_attention_load_mxfp6e3m2_serial_(row, row + depth, index, 0)
                                                 : ((nk_f32_t const *)row)[index];
}

/** @c nk_attention_score_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE nk_f32_t nk_attention_score_mxfp6e3m2_serial_(nk_u8_t const *query_row, nk_u8_t const *query_scales,
                                                             nk_u8_t const *key_row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_f32_t scale2) {
    nk_cross_wide_sum_t dot = {0, 0};
    if (exponent == nk_attention_raw_plane_k_) {
        nk_cross_wide_sum_t query_sumsq, key_sumsq;
        dot = nk_cross_scaled_exact_wide_mxfp6e3m2_serial_(query_row, query_scales, key_row, key_row + depth, depth,
                                                           &query_sumsq, &key_sumsq);
    }
    else
        for (nk_size_t block = 0; block * 32 < depth; block++) {
            nk_i32_t query_exponent;
            nk_f32_t const mantissa = nk_ue8m0_split_serial_(query_scales[block], &query_exponent);
            nk_f32_t block_sum = 0;
            for (nk_size_t index = block * 32; index != (block + 1) * 32; index++)
                block_sum += nk_e3m2_load_f32_serial_(query_row, index) * ((nk_f32_t const *)key_row)[index];
            nk_cross_wide_add_serial_(&dot, block_sum * mantissa,
                                      query_exponent + exponent - nk_cross_scaled_headroom_k);
        }
    return nk_scale_f32_serial_(dot.sum * scale2, dot.exponent);
}

/** @c nk_attention_pack_plane_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE nk_i8_t nk_attention_pack_plane_mxfp6e3m2_serial_(nk_mxfp6e3m2_cref_t const *operand, nk_size_t stride,
                                                                 nk_size_t first_position, nk_size_t positions,
                                                                 nk_size_t head_idx, nk_size_t depth, nk_f32_t *plane) {
    nk_size_t const scales_stride = stride / 32, blocks = depth / 32, codes_bytes = depth;
    nk_u8_t const *codes = (nk_u8_t const *)operand->elements + first_position * stride + head_idx * codes_bytes;
    nk_u8_t const *scales = (nk_u8_t const *)operand->scales + first_position * scales_stride + head_idx * blocks;
    nk_i8_t const exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, positions, blocks);
    nk_i32_t const base = nk_attention_plane_base_serial_(exponent);
    for (nk_size_t position_idx = 0; position_idx < positions; position_idx++) {
        nk_u8_t const *row_codes = codes + position_idx * stride, *row_scales = scales + position_idx * scales_stride;
        nk_f32_t *row = plane + position_idx * depth;
        if (exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_row_serial_(row_codes, codes_bytes, row_scales, blocks, depth * sizeof(nk_f32_t), row);
        else
            for (nk_size_t index = 0; index < depth; index++)
                row[index] = nk_attention_load_mxfp6e3m2_serial_(row_codes, row_scales, index, base);
    }
    return exponent;
}

/** Element @p index of an MXFP8 E4M3 row of @p codes times its block scale, rebased by @p base. */
NUMKONG_INLINE nk_f32_t nk_attention_load_mxfp8e4m3_serial_(nk_u8_t const *codes, nk_u8_t const *scales,
                                                            nk_size_t index, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_serial_(scales[index / 32], &exponent);
    return nk_scale_f32_serial_(nk_e4m3_load_f32_serial_(codes, index) * mantissa, exponent - base);
}

/** @c nk_attention_plane_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE nk_f32_t nk_attention_plane_mxfp8e4m3_serial_(nk_u8_t const *row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_size_t index) {
    return exponent == nk_attention_raw_plane_k_ ? nk_attention_load_mxfp8e4m3_serial_(row, row + depth, index, 0)
                                                 : ((nk_f32_t const *)row)[index];
}

/** @c nk_attention_score_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE nk_f32_t nk_attention_score_mxfp8e4m3_serial_(nk_u8_t const *query_row, nk_u8_t const *query_scales,
                                                             nk_u8_t const *key_row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_f32_t scale2) {
    nk_cross_wide_sum_t dot = {0, 0};
    if (exponent == nk_attention_raw_plane_k_) {
        nk_cross_wide_sum_t query_sumsq, key_sumsq;
        dot = nk_cross_scaled_exact_wide_mxfp8e4m3_serial_(query_row, query_scales, key_row, key_row + depth, depth,
                                                           &query_sumsq, &key_sumsq);
    }
    else
        for (nk_size_t block = 0; block * 32 < depth; block++) {
            nk_i32_t query_exponent;
            nk_f32_t const mantissa = nk_ue8m0_split_serial_(query_scales[block], &query_exponent);
            nk_f32_t block_sum = 0;
            for (nk_size_t index = block * 32; index != (block + 1) * 32; index++)
                block_sum += nk_e4m3_load_f32_serial_(query_row, index) * ((nk_f32_t const *)key_row)[index];
            nk_cross_wide_add_serial_(&dot, block_sum * mantissa,
                                      query_exponent + exponent - nk_cross_scaled_headroom_k);
        }
    return nk_scale_f32_serial_(dot.sum * scale2, dot.exponent);
}

/** @c nk_attention_pack_plane_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE nk_i8_t nk_attention_pack_plane_mxfp8e4m3_serial_(nk_mxfp8e4m3_cref_t const *operand, nk_size_t stride,
                                                                 nk_size_t first_position, nk_size_t positions,
                                                                 nk_size_t head_idx, nk_size_t depth, nk_f32_t *plane) {
    nk_size_t const scales_stride = stride / 32, blocks = depth / 32, codes_bytes = depth;
    nk_u8_t const *codes = (nk_u8_t const *)operand->elements + first_position * stride + head_idx * codes_bytes;
    nk_u8_t const *scales = (nk_u8_t const *)operand->scales + first_position * scales_stride + head_idx * blocks;
    nk_i8_t const exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, positions, blocks);
    nk_i32_t const base = nk_attention_plane_base_serial_(exponent);
    for (nk_size_t position_idx = 0; position_idx < positions; position_idx++) {
        nk_u8_t const *row_codes = codes + position_idx * stride, *row_scales = scales + position_idx * scales_stride;
        nk_f32_t *row = plane + position_idx * depth;
        if (exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_row_serial_(row_codes, codes_bytes, row_scales, blocks, depth * sizeof(nk_f32_t), row);
        else
            for (nk_size_t index = 0; index < depth; index++)
                row[index] = nk_attention_load_mxfp8e4m3_serial_(row_codes, row_scales, index, base);
    }
    return exponent;
}

/** Element @p index of an MXFP8 E5M2 row of @p codes times its block scale, rebased by @p base. */
NUMKONG_INLINE nk_f32_t nk_attention_load_mxfp8e5m2_serial_(nk_u8_t const *codes, nk_u8_t const *scales,
                                                            nk_size_t index, nk_i32_t base) {
    nk_i32_t exponent;
    nk_f32_t const mantissa = nk_ue8m0_split_serial_(scales[index / 32], &exponent);
    return nk_scale_f32_serial_(nk_e5m2_load_f32_serial_(codes, index) * mantissa, exponent - base);
}

/** @c nk_attention_plane_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE nk_f32_t nk_attention_plane_mxfp8e5m2_serial_(nk_u8_t const *row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_size_t index) {
    return exponent == nk_attention_raw_plane_k_ ? nk_attention_load_mxfp8e5m2_serial_(row, row + depth, index, 0)
                                                 : ((nk_f32_t const *)row)[index];
}

/** @c nk_attention_score_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE nk_f32_t nk_attention_score_mxfp8e5m2_serial_(nk_u8_t const *query_row, nk_u8_t const *query_scales,
                                                             nk_u8_t const *key_row, nk_i8_t exponent, nk_size_t depth,
                                                             nk_f32_t scale2) {
    nk_cross_wide_sum_t dot = {0, 0};
    if (exponent == nk_attention_raw_plane_k_) {
        nk_cross_wide_sum_t query_sumsq, key_sumsq;
        dot = nk_cross_scaled_exact_wide_mxfp8e5m2_serial_(query_row, query_scales, key_row, key_row + depth, depth,
                                                           &query_sumsq, &key_sumsq);
    }
    else
        for (nk_size_t block = 0; block * 32 < depth; block++) {
            nk_i32_t query_exponent;
            nk_f32_t const mantissa = nk_ue8m0_split_serial_(query_scales[block], &query_exponent);
            nk_f32_t block_sum = 0;
            for (nk_size_t index = block * 32; index != (block + 1) * 32; index++)
                block_sum += nk_e5m2_load_f32_serial_(query_row, index) * ((nk_f32_t const *)key_row)[index];
            nk_cross_wide_add_serial_(&dot, block_sum * mantissa,
                                      query_exponent + exponent - nk_cross_scaled_headroom_k);
        }
    return nk_scale_f32_serial_(dot.sum * scale2, dot.exponent);
}

/** @c nk_attention_pack_plane_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE nk_i8_t nk_attention_pack_plane_mxfp8e5m2_serial_(nk_mxfp8e5m2_cref_t const *operand, nk_size_t stride,
                                                                 nk_size_t first_position, nk_size_t positions,
                                                                 nk_size_t head_idx, nk_size_t depth, nk_f32_t *plane) {
    nk_size_t const scales_stride = stride / 32, blocks = depth / 32, codes_bytes = depth;
    nk_u8_t const *codes = (nk_u8_t const *)operand->elements + first_position * stride + head_idx * codes_bytes;
    nk_u8_t const *scales = (nk_u8_t const *)operand->scales + first_position * scales_stride + head_idx * blocks;
    nk_i8_t const exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, positions, blocks);
    nk_i32_t const base = nk_attention_plane_base_serial_(exponent);
    for (nk_size_t position_idx = 0; position_idx < positions; position_idx++) {
        nk_u8_t const *row_codes = codes + position_idx * stride, *row_scales = scales + position_idx * scales_stride;
        nk_f32_t *row = plane + position_idx * depth;
        if (exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_row_serial_(row_codes, codes_bytes, row_scales, blocks, depth * sizeof(nk_f32_t), row);
        else
            for (nk_size_t index = 0; index < depth; index++)
                row[index] = nk_attention_load_mxfp8e5m2_serial_(row_codes, row_scales, index, base);
    }
    return exponent;
}

#if NUMKONG_TARGET_SERIAL
/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_segments(void const *key_value_packed, nk_size_t segment_count,
                                                     nk_u32_t const **key_offsets, nk_u32_t const **key_lengths) {
    *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_serial(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_serial(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_serial(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_serial(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_serial(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * sizeof(nk_f32_t)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/** BF16 packing core: widen K and V rows to F32 planes `[key_value_head][position][channel]`, and
 *  record the packing @p capability. */
NUMKONG_INLINE void nk_attention_pack_bf16_serial_(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride,
                                                   nk_size_t value_stride, void *key_value_packed,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_f32_t *values_plane = keys_plane + key_value_head_count * plane_floats;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * sizeof(nk_bf16_t);
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * sizeof(nk_bf16_t);
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = nk_attention_load_bf16_serial_(
                    keys_row + channel_idx * sizeof(nk_bf16_t));
                values_plane[position_idx * depth + channel_idx] = nk_attention_load_bf16_serial_(
                    values_row + channel_idx * sizeof(nk_bf16_t));
            }
        }
    }
}

/** F16 packing core: widen K and V rows to F32 planes `[key_value_head][position][channel]`, and
 *  record the packing @p capability. */
NUMKONG_INLINE void nk_attention_pack_f16_serial_(nk_f16_t const *keys, nk_f16_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,
                                                  void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_f32_t *values_plane = keys_plane + key_value_head_count * plane_floats;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * sizeof(nk_f16_t);
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * sizeof(nk_f16_t);
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = nk_attention_load_f16_serial_(
                    keys_row + channel_idx * sizeof(nk_f16_t));
                values_plane[position_idx * depth + channel_idx] = nk_attention_load_f16_serial_(
                    values_row + channel_idx * sizeof(nk_f16_t));
            }
        }
    }
}

/** E4M3 packing core: widen K and V rows to F32 planes `[key_value_head][position][channel]`, and
 *  record the packing @p capability. */
NUMKONG_INLINE void nk_attention_pack_e4m3_serial_(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride,
                                                   nk_size_t value_stride, void *key_value_packed,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_f32_t *values_plane = keys_plane + key_value_head_count * plane_floats;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * sizeof(nk_e4m3_t);
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * sizeof(nk_e4m3_t);
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = nk_attention_load_e4m3_serial_(
                    keys_row + channel_idx * sizeof(nk_e4m3_t));
                values_plane[position_idx * depth + channel_idx] = nk_attention_load_e4m3_serial_(
                    values_row + channel_idx * sizeof(nk_e4m3_t));
            }
        }
    }
}

/** I8 packing core: copy K and V rows into raw I8 planes `[key_value_head][position][channel]`, and
 *  record @p capability as the packing capability. */
NUMKONG_INLINE void nk_attention_pack_i8_serial_(                                                //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,           //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, //
    nk_size_t tasks_end, nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth, capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth;
        nk_i8_t *keys_plane = (nk_i8_t *)(payload_base + payload_offset) + key_value_head_idx * plane_bytes;
        nk_i8_t *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = (nk_i8_t)keys_row[channel_idx];
                values_plane[position_idx * depth + channel_idx] = (nk_i8_t)values_row[channel_idx];
            }
        }
    }
}

/** NVFP4 packing core: K and V planes of each element times its block scale, and both tensor
 *  scales in the header, which the window from task 0 writes. */
NUMKONG_INLINE void nk_attention_pack_nvfp4_serial_(nk_nvfp4_cref_t const *keys, nk_nvfp4_cref_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    if (tasks_begin == 0) {
        nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
        header->key_tensor_scale = nk_cross_tensor_scale_serial_(keys->tensor_scale);
        header->value_tensor_scale = nk_cross_tensor_scale_serial_(values->tensor_scale);
    }
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const key_scales_stride = key_stride / 8, value_scales_stride = value_stride / 8;

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_f32_t *values_plane = keys_plane + key_value_head_count * plane_floats;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            nk_size_t const position = position_first + position_idx;
            nk_u8_t const *keys_row = (nk_u8_t const *)keys->elements + position * key_stride +
                                      key_value_head_idx * depth / 2;
            nk_u8_t const *values_row = (nk_u8_t const *)values->elements + position * value_stride +
                                        key_value_head_idx * depth / 2;
            nk_u8_t const *key_scales = (nk_u8_t const *)keys->scales + position * key_scales_stride +
                                        key_value_head_idx * depth / 16;
            nk_u8_t const *value_scales = (nk_u8_t const *)values->scales + position * value_scales_stride +
                                          key_value_head_idx * depth / 16;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = nk_attention_load_nvfp4_serial_(keys_row, key_scales,
                                                                                                 channel_idx);
                values_plane[position_idx * depth + channel_idx] = nk_attention_load_nvfp4_serial_(
                    values_row, value_scales, channel_idx);
            }
        }
    }
}

/** MXFP4 packing core: K and V planes through @c nk_attention_pack_plane_mxfp4_serial_, and their
 *  entries in the plane-exponent table, whose padding the window from task 0 zeroes. */
NUMKONG_INLINE void nk_attention_pack_mxfp4_serial_(nk_mxfp4_cref_t const *keys, nk_mxfp4_cref_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, 1, depth * sizeof(nk_f32_t),
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_i8_t *exponents = plane_exponents + 2 * segment_idx * key_value_head_count + key_value_head_idx;
        exponents[0] = nk_attention_pack_plane_mxfp4_serial_(keys, key_stride, key_offsets[segment_idx], position_count,
                                                             key_value_head_idx, depth, keys_plane);
        exponents[key_value_head_count] = nk_attention_pack_plane_mxfp4_serial_(
            values, value_stride, key_offsets[segment_idx], position_count, key_value_head_idx, depth,
            keys_plane + key_value_head_count * plane_floats);
    }
}

/** @c nk_attention_pack_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE void nk_attention_pack_mxfp6e2m3_serial_(
    nk_mxfp6e2m3_cref_t const *keys, nk_mxfp6e2m3_cref_t const *values, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, 1, depth * sizeof(nk_f32_t),
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_i8_t *exponents = plane_exponents + 2 * segment_idx * key_value_head_count + key_value_head_idx;
        exponents[0] = nk_attention_pack_plane_mxfp6e2m3_serial_(keys, key_stride, key_offsets[segment_idx],
                                                                 position_count, key_value_head_idx, depth, keys_plane);
        exponents[key_value_head_count] = nk_attention_pack_plane_mxfp6e2m3_serial_(
            values, value_stride, key_offsets[segment_idx], position_count, key_value_head_idx, depth,
            keys_plane + key_value_head_count * plane_floats);
    }
}

/** @c nk_attention_pack_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE void nk_attention_pack_mxfp6e3m2_serial_(
    nk_mxfp6e3m2_cref_t const *keys, nk_mxfp6e3m2_cref_t const *values, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, 1, depth * sizeof(nk_f32_t),
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_i8_t *exponents = plane_exponents + 2 * segment_idx * key_value_head_count + key_value_head_idx;
        exponents[0] = nk_attention_pack_plane_mxfp6e3m2_serial_(keys, key_stride, key_offsets[segment_idx],
                                                                 position_count, key_value_head_idx, depth, keys_plane);
        exponents[key_value_head_count] = nk_attention_pack_plane_mxfp6e3m2_serial_(
            values, value_stride, key_offsets[segment_idx], position_count, key_value_head_idx, depth,
            keys_plane + key_value_head_count * plane_floats);
    }
}

/** @c nk_attention_pack_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE void nk_attention_pack_mxfp8e4m3_serial_(
    nk_mxfp8e4m3_cref_t const *keys, nk_mxfp8e4m3_cref_t const *values, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, 1, depth * sizeof(nk_f32_t),
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_i8_t *exponents = plane_exponents + 2 * segment_idx * key_value_head_count + key_value_head_idx;
        exponents[0] = nk_attention_pack_plane_mxfp8e4m3_serial_(keys, key_stride, key_offsets[segment_idx],
                                                                 position_count, key_value_head_idx, depth, keys_plane);
        exponents[key_value_head_count] = nk_attention_pack_plane_mxfp8e4m3_serial_(
            values, value_stride, key_offsets[segment_idx], position_count, key_value_head_idx, depth,
            keys_plane + key_value_head_count * plane_floats);
    }
}

/** @c nk_attention_pack_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE void nk_attention_pack_mxfp8e5m2_serial_(
    nk_mxfp8e5m2_cref_t const *keys, nk_mxfp8e5m2_cref_t const *values, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capability) {

    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth * sizeof(nk_f32_t), capability);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, 1, depth * sizeof(nk_f32_t),
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth * sizeof(nk_f32_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offset) + key_value_head_idx * plane_floats;
        nk_i8_t *exponents = plane_exponents + 2 * segment_idx * key_value_head_count + key_value_head_idx;
        exponents[0] = nk_attention_pack_plane_mxfp8e5m2_serial_(keys, key_stride, key_offsets[segment_idx],
                                                                 position_count, key_value_head_idx, depth, keys_plane);
        exponents[key_value_head_count] = nk_attention_pack_plane_mxfp8e5m2_serial_(
            values, value_stride, key_offsets[segment_idx], position_count, key_value_head_idx, depth,
            keys_plane + key_value_head_count * plane_floats);
    }
}

#if NUMKONG_TARGET_SERIAL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_attention_pack_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_bf16_t const *keys,
                                                      nk_size_t key_stride, nk_bf16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_bf16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                   key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_f16_t const *keys,
                                                     nk_size_t key_stride, nk_f16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_f16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_e4m3_t const *keys,
                                                      nk_size_t key_stride, nk_e4m3_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                   key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_nvfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                       nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 16 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_nvfp4_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                    key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                    nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                       nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_mxfp4_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                    key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                    nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_mxfp6e2m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_mxfp6e3m2_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_mxfp8e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_mxfp8e5m2_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

/**
 *  @brief BF16 attention core: exact two-sweep softmax attention per (query token, head) task.
 *
 *  Per query row: sweep 1 finds the row maximum of score · scale₂; sweep 2 recomputes the scores,
 *  accumulating V rows weighted by 2^(score · scale₂ − max₂) straight into the output row, used as
 *  the accumulator, then normalizes by the accumulated sum. With no scratch, @p depth and
 *  @c position_count are unbounded. Recomputing scores costs ~1.5× the arithmetic of a buffered
 *  implementation and buys exact width-agnosticism with zero allocations. Row @c r reads only the
 *  keys that @p band shows to its position, and when @p log_sum_exp is not null, it receives the
 *  row's natural log of Σ exp(score · scale).
 */
NUMKONG_INLINE void nk_attention_packed_bf16_serial_(nk_bf16_t const *queries, void const *key_value_packed,
                                                     nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                     nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_floats = position_count * depth;
            nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                         (head_idx / head_group_size) * plane_floats;
            nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                char const *query_row = (char const *)queries + (query_first + row_idx) * query_stride +
                                        head_idx * depth * sizeof(nk_bf16_t);
                nk_f32_t *output_row = output + (query_first + row_idx) * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_bf16_serial_(query_row + channel_idx * sizeof(nk_bf16_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const scaled2 = score * scale2;
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_bf16_serial_(query_row + channel_idx * sizeof(nk_bf16_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - max2);
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * values_plane[position_idx * depth + channel_idx];
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] *= inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[(query_first + row_idx) * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        max2, weights_sum);
            }
        }
    }
}

/**
 *  @brief F16 attention core: exact two-sweep softmax attention per (query token, head) task.
 *
 *  Per query row: sweep 1 finds the row maximum of score · scale₂; sweep 2 recomputes the scores,
 *  accumulating V rows weighted by 2^(score · scale₂ − max₂) straight into the output row, used as
 *  the accumulator, then normalizes by the accumulated sum. With no scratch, @p depth and
 *  @c position_count are unbounded. Recomputing scores costs ~1.5× the arithmetic of a buffered
 *  implementation and buys exact width-agnosticism with zero allocations. Row @c r reads only the
 *  keys that @p band shows to its position, and when @p log_sum_exp is not null, it receives the
 *  row's natural log of Σ exp(score · scale).
 */
NUMKONG_INLINE void nk_attention_packed_f16_serial_(nk_f16_t const *queries, void const *key_value_packed,
                                                    nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                    nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_floats = position_count * depth;
            nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                         (head_idx / head_group_size) * plane_floats;
            nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                char const *query_row = (char const *)queries + (query_first + row_idx) * query_stride +
                                        head_idx * depth * sizeof(nk_f16_t);
                nk_f32_t *output_row = output + (query_first + row_idx) * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_f16_serial_(query_row + channel_idx * sizeof(nk_f16_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const scaled2 = score * scale2;
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_f16_serial_(query_row + channel_idx * sizeof(nk_f16_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - max2);
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * values_plane[position_idx * depth + channel_idx];
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] *= inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[(query_first + row_idx) * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        max2, weights_sum);
            }
        }
    }
}

/**
 *  @brief E4M3 attention core: exact two-sweep softmax attention per (query token, head) task.
 *
 *  Per query row: sweep 1 finds the row maximum of score · scale₂; sweep 2 recomputes the scores,
 *  accumulating V rows weighted by 2^(score · scale₂ − max₂) straight into the output row, used as
 *  the accumulator, then normalizes by the accumulated sum. With no scratch, @p depth and
 *  @c position_count are unbounded. Recomputing scores costs ~1.5× the arithmetic of a buffered
 *  implementation and buys exact width-agnosticism with zero allocations. Row @c r reads only the
 *  keys that @p band shows to its position, and when @p log_sum_exp is not null, it receives the
 *  row's natural log of Σ exp(score · scale).
 */
NUMKONG_INLINE void nk_attention_packed_e4m3_serial_(nk_e4m3_t const *queries, void const *key_value_packed,
                                                     nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                     nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_floats = position_count * depth;
            nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                         (head_idx / head_group_size) * plane_floats;
            nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                char const *query_row = (char const *)queries + (query_first + row_idx) * query_stride +
                                        head_idx * depth * sizeof(nk_e4m3_t);
                nk_f32_t *output_row = output + (query_first + row_idx) * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_e4m3_serial_(query_row + channel_idx * sizeof(nk_e4m3_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const scaled2 = score * scale2;
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_e4m3_serial_(query_row + channel_idx * sizeof(nk_e4m3_t)) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - max2);
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * values_plane[position_idx * depth + channel_idx];
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] *= inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[(query_first + row_idx) * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        max2, weights_sum);
            }
        }
    }
}

/** I8 attention core: exact I32 scores, U8-quantized weights, same row ranges and outputs as
 *  @c nk_attention_serial_. */
NUMKONG_INLINE void nk_attention_packed_i8_serial_(                                                 //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,  //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * depth;
            nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        (head_idx / head_group_size) * plane_bytes;
            nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                             (query_first + row_idx) * query_stride) +
                                           head_idx * depth;
                nk_f32_t *output_row = output + (query_first + row_idx) * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN; // scores are exact I32 integer dots; row max found before quantizing
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_i32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += (nk_i32_t)query_row[channel_idx] *
                                 (nk_i32_t)keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const scaled2 = (nk_f32_t)score * scale2;
                    if (scaled2 > max2) max2 = scaled2;
                }
                nk_f32_t sum_weights = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_i32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += (nk_i32_t)query_row[channel_idx] *
                                 (nk_i32_t)keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const weight_f32 = nk_f32_exp2_serial_((nk_f32_t)score * scale2 - max2);
                    nk_u32_t const weight_u8 = (nk_u32_t)(weight_f32 * 255.0f + 0.5f);
                    if (weight_u8 == 0) continue;
                    sum_weights += (nk_f32_t)weight_u8;
                    nk_f32_t const weight = (nk_f32_t)weight_u8;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * (nk_f32_t)values_plane[position_idx * depth + channel_idx];
                }
                nk_f32_t const inverse_sum = sum_weights > 0 ? 1 / sum_weights : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] *= inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[(query_first + row_idx) * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        max2, sum_weights / 255.0f);
            }
        }
    }
}

/** NVFP4 attention core: @c nk_attention_packed_bf16_serial_ over decoded queries, with the query
 *  and key tensor scales in the score multiplier, and the value one in the normalization. */
NUMKONG_INLINE void nk_attention_packed_nvfp4_serial_(nk_nvfp4_cref_t const *queries, void const *key_value_packed,
                                                      nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                      nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 16 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 8;
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(
        nk_cross_tensor_scale_serial_(queries->tensor_scale), header->key_tensor_scale);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_ * factor.mantissa;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_floats = position_count * depth;
            nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                         (head_idx / head_group_size) * plane_floats;
            nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth / 2;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 16;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_nvfp4_serial_(query_row, query_row_scales, channel_idx) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const scaled2 = nk_scale_f32_serial_(score * scale2, factor.exponent);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t score = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        score += nk_attention_load_nvfp4_serial_(query_row, query_row_scales, channel_idx) *
                                 keys_plane[position_idx * depth + channel_idx];
                    nk_f32_t const weight = nk_f32_exp2_serial_(nk_scale_f32_serial_(score * scale2, factor.exponent) -
                                                                max2);
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * values_plane[position_idx * depth + channel_idx];
                }
                nk_f32_t const normalization = weights_sum > 0 ? header->value_tensor_scale / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] *= normalization;
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/**
 *  @brief MXFP4 attention core: @c nk_attention_packed_bf16_serial_ over the planes of
 *      @c nk_attention_pack_mxfp4_serial_.
 *
 *  Scores come from @c nk_attention_score_mxfp4_serial_, so queries of any scales round once. Each
 *  row accumulates value rows under their plane's base and applies 2 to that base along with the
 *  normalization, rounding once per output.
 */
NUMKONG_INLINE void nk_attention_packed_mxfp4_serial_(nk_mxfp4_cref_t const *queries, void const *key_value_packed,
                                                      nk_f32_t *output, nk_f32_t *log_sum_exp, nk_size_t head_count,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *query_offsets, nk_size_t query_stride,
                                                      nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 16, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        key_value_head_idx * position_count * row_bytes;
            nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth / 2;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const scaled2 = nk_attention_score_mxfp4_serial_(query_row, query_row_scales,
                                                                              keys_plane + position_idx * row_bytes,
                                                                              key_exponent, depth, scale2);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp4_serial_(query_row, query_row_scales,
                                                         keys_plane + position_idx * row_bytes, key_exponent, depth,
                                                         scale2) -
                        max2);
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * nk_attention_plane_mxfp4_serial_(value_row, value_exponent,
                                                                                             depth, channel_idx);
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] = nk_scale_f32_serial_(output_row[channel_idx] * inverse_sum, value_base);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE void nk_attention_packed_mxfp6e2m3_serial_(
    nk_mxfp6e2m3_cref_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        key_value_head_idx * position_count * row_bytes;
            nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const scaled2 = nk_attention_score_mxfp6e2m3_serial_(query_row, query_row_scales,
                                                                                  keys_plane + position_idx * row_bytes,
                                                                                  key_exponent, depth, scale2);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp6e2m3_serial_(query_row, query_row_scales,
                                                             keys_plane + position_idx * row_bytes, key_exponent, depth,
                                                             scale2) -
                        max2);
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * nk_attention_plane_mxfp6e2m3_serial_(
                                                                value_row, value_exponent, depth, channel_idx);
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] = nk_scale_f32_serial_(output_row[channel_idx] * inverse_sum, value_base);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE void nk_attention_packed_mxfp6e3m2_serial_(
    nk_mxfp6e3m2_cref_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        key_value_head_idx * position_count * row_bytes;
            nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const scaled2 = nk_attention_score_mxfp6e3m2_serial_(query_row, query_row_scales,
                                                                                  keys_plane + position_idx * row_bytes,
                                                                                  key_exponent, depth, scale2);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp6e3m2_serial_(query_row, query_row_scales,
                                                             keys_plane + position_idx * row_bytes, key_exponent, depth,
                                                             scale2) -
                        max2);
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * nk_attention_plane_mxfp6e3m2_serial_(
                                                                value_row, value_exponent, depth, channel_idx);
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] = nk_scale_f32_serial_(output_row[channel_idx] * inverse_sum, value_base);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE void nk_attention_packed_mxfp8e4m3_serial_(
    nk_mxfp8e4m3_cref_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        key_value_head_idx * position_count * row_bytes;
            nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const scaled2 = nk_attention_score_mxfp8e4m3_serial_(query_row, query_row_scales,
                                                                                  keys_plane + position_idx * row_bytes,
                                                                                  key_exponent, depth, scale2);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp8e4m3_serial_(query_row, query_row_scales,
                                                             keys_plane + position_idx * row_bytes, key_exponent, depth,
                                                             scale2) -
                        max2);
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * nk_attention_plane_mxfp8e4m3_serial_(
                                                                value_row, value_exponent, depth, channel_idx);
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] = nk_scale_f32_serial_(output_row[channel_idx] * inverse_sum, value_base);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE void nk_attention_packed_mxfp8e5m2_serial_(
    nk_mxfp8e5m2_cref_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;
    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                        key_value_head_idx * position_count * row_bytes;
            nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t const token = query_first + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                nk_f32_t max2 = NUMKONG_F32_MIN;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const scaled2 = nk_attention_score_mxfp8e5m2_serial_(query_row, query_row_scales,
                                                                                  keys_plane + position_idx * row_bytes,
                                                                                  key_exponent, depth, scale2);
                    if (scaled2 > max2) max2 = scaled2;
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
                nk_f32_t weights_sum = 0;
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp8e5m2_serial_(query_row, query_row_scales,
                                                             keys_plane + position_idx * row_bytes, key_exponent, depth,
                                                             scale2) -
                        max2);
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    weights_sum += weight;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        output_row[channel_idx] += weight * nk_attention_plane_mxfp8e5m2_serial_(
                                                                value_row, value_exponent, depth, channel_idx);
                }
                nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] = nk_scale_f32_serial_(output_row[channel_idx] * inverse_sum, value_base);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2, weights_sum);
            }
        }
    }
}

/**
 *  @brief BF16 attention backward: per (segment, kv_head) task, the gradients of exact softmax
 *      attention, recomputing every weight from its row's log-sum-exp.
 *
 *  Zeroes the task's key and value gradient rows, then walks the query heads sharing the KV head
 *  and their rows in order, so the sums run the same way on every call. Each row takes D = dO · O,
 *  and each visible key P = 2^(score · scale₂ − lse₂) and dV += P · dO, then adds dS · K to dQ and
 *  dS · Q to dK for the score gradient dS = P · (dO · V − D) · scale.
 */
NUMKONG_INLINE void nk_attention_backward_bf16_serial_(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                     key_value_head_idx * plane_floats;
        nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                char const *query_row = (char const *)queries + token * query_stride +
                                        head_idx * depth * sizeof(nk_bf16_t);
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const *key_row = keys_plane + position_idx * depth;
                    nk_f32_t const *value_row = values_plane + position_idx * depth;
                    nk_f32_t score = 0, weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        score += nk_attention_load_bf16_serial_(query_row + channel_idx * sizeof(nk_bf16_t)) *
                                 key_row[channel_idx];
                        weight_gradient += output_gradient_row[channel_idx] * value_row[channel_idx];
                    }
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - log_sum_exp2);
                    nk_f32_t const score_gradient = weight * (weight_gradient - row_dot) * scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient * key_row[channel_idx];
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_bf16_serial_(
                                                             query_row + channel_idx * sizeof(nk_bf16_t));
                    }
                }
            }
        }
    }
}

/**
 *  @brief F16 attention backward: per (segment, kv_head) task, the gradients of exact softmax
 *      attention, recomputing every weight from its row's log-sum-exp.
 *
 *  Zeroes the task's key and value gradient rows, then walks the query heads sharing the KV head
 *  and their rows in order, so the sums run the same way on every call. Each row takes D = dO · O,
 *  and each visible key P = 2^(score · scale₂ − lse₂) and dV += P · dO, then adds dS · K to dQ and
 *  dS · Q to dK for the score gradient dS = P · (dO · V − D) · scale.
 */
NUMKONG_INLINE void nk_attention_backward_f16_serial_(
    nk_f16_t const *queries, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                     key_value_head_idx * plane_floats;
        nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                char const *query_row = (char const *)queries + token * query_stride +
                                        head_idx * depth * sizeof(nk_f16_t);
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const *key_row = keys_plane + position_idx * depth;
                    nk_f32_t const *value_row = values_plane + position_idx * depth;
                    nk_f32_t score = 0, weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        score += nk_attention_load_f16_serial_(query_row + channel_idx * sizeof(nk_f16_t)) *
                                 key_row[channel_idx];
                        weight_gradient += output_gradient_row[channel_idx] * value_row[channel_idx];
                    }
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - log_sum_exp2);
                    nk_f32_t const score_gradient = weight * (weight_gradient - row_dot) * scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient * key_row[channel_idx];
                        key_gradient_row[channel_idx] +=
                            score_gradient * nk_attention_load_f16_serial_(query_row + channel_idx * sizeof(nk_f16_t));
                    }
                }
            }
        }
    }
}

/**
 *  @brief E4M3 attention backward: per (segment, kv_head) task, the gradients of exact softmax
 *      attention, recomputing every weight from its row's log-sum-exp.
 *
 *  Zeroes the task's key and value gradient rows, then walks the query heads sharing the KV head
 *  and their rows in order, so the sums run the same way on every call. Each row takes D = dO · O,
 *  and each visible key P = 2^(score · scale₂ − lse₂) and dV += P · dO, then adds dS · K to dQ and
 *  dS · Q to dK for the score gradient dS = P · (dO · V − D) · scale.
 */
NUMKONG_INLINE void nk_attention_backward_e4m3_serial_(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                     key_value_head_idx * plane_floats;
        nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                char const *query_row = (char const *)queries + token * query_stride +
                                        head_idx * depth * sizeof(nk_e4m3_t);
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const *key_row = keys_plane + position_idx * depth;
                    nk_f32_t const *value_row = values_plane + position_idx * depth;
                    nk_f32_t score = 0, weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        score += nk_attention_load_e4m3_serial_(query_row + channel_idx * sizeof(nk_e4m3_t)) *
                                 key_row[channel_idx];
                        weight_gradient += output_gradient_row[channel_idx] * value_row[channel_idx];
                    }
                    nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - log_sum_exp2);
                    nk_f32_t const score_gradient = weight * (weight_gradient - row_dot) * scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient * key_row[channel_idx];
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_e4m3_serial_(
                                                             query_row + channel_idx * sizeof(nk_e4m3_t));
                    }
                }
            }
        }
    }
}

/** NVFP4 attention backward: @c nk_attention_backward_bf16_serial_ over decoded queries, with every
 *  tensor scale applied where its operand enters. */
NUMKONG_INLINE void nk_attention_backward_nvfp4_serial_(
    nk_nvfp4_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 16 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 8;
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const query_tensor_scale = nk_cross_tensor_scale_serial_(queries->tensor_scale);
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(query_tensor_scale,
                                                                           header->key_tensor_scale);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_ * factor.mantissa;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                     key_value_head_idx * plane_floats;
        nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth / 2;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 16;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_f32_t const *key_row = keys_plane + position_idx * depth;
                    nk_f32_t const *value_row = values_plane + position_idx * depth;
                    nk_f32_t score = 0, weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        score += nk_attention_load_nvfp4_serial_(query_row, query_row_scales, channel_idx) *
                                 key_row[channel_idx];
                        weight_gradient += output_gradient_row[channel_idx] * value_row[channel_idx];
                    }
                    nk_f32_t const weight = nk_f32_exp2_serial_(nk_scale_f32_serial_(score * scale2, factor.exponent) -
                                                                log_sum_exp2);
                    nk_f32_t const score_gradient = weight * (weight_gradient * header->value_tensor_scale - row_dot) *
                                                    scale;
                    nk_f32_t const query_score_gradient = score_gradient * query_tensor_scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient * key_row[channel_idx];
                        key_gradient_row[channel_idx] += query_score_gradient *
                                                         nk_attention_load_nvfp4_serial_(query_row, query_row_scales,
                                                                                         channel_idx);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] *= header->key_tensor_scale;
            }
        }
    }
}

/** MXFP4 attention backward: @c nk_attention_backward_bf16_serial_ over the planes of
 *  @c nk_attention_pack_mxfp4_serial_. Value dots take 2 to the value plane's base, and each query
 *  gradient row sums under the key plane's base, which it applies once at the end. */
NUMKONG_INLINE void nk_attention_backward_mxfp4_serial_(
    nk_mxfp4_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 16, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    key_value_head_idx * position_count * row_bytes;
        nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
        nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
        nk_i8_t const value_exponent =
            plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
        nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent),
                       value_base = nk_attention_plane_base_serial_(value_exponent);
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth / 2;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_u8_t const *key_row = keys_plane + position_idx * row_bytes;
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    nk_f32_t weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        weight_gradient += output_gradient_row[channel_idx] *
                                           nk_attention_plane_mxfp4_serial_(value_row, value_exponent, depth,
                                                                            channel_idx);
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp4_serial_(query_row, query_row_scales, key_row, key_exponent, depth,
                                                         scale2) -
                        log_sum_exp2);
                    nk_f32_t const score_gradient = weight *
                                                    (nk_scale_f32_serial_(weight_gradient, value_base) - row_dot) *
                                                    scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient *
                                                           nk_attention_plane_mxfp4_serial_(key_row, key_exponent,
                                                                                            depth, channel_idx);
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_mxfp4_serial_(query_row, query_row_scales,
                                                                                         channel_idx, 0);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] = nk_scale_f32_serial_(query_gradient_row[channel_idx], key_base);
            }
        }
    }
}

/** @c nk_attention_backward_mxfp4_serial_ for MXFP6 E2M3. */
NUMKONG_INLINE void nk_attention_backward_mxfp6e2m3_serial_(
    nk_mxfp6e2m3_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    key_value_head_idx * position_count * row_bytes;
        nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
        nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
        nk_i8_t const value_exponent =
            plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
        nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent),
                       value_base = nk_attention_plane_base_serial_(value_exponent);
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_u8_t const *key_row = keys_plane + position_idx * row_bytes;
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    nk_f32_t weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        weight_gradient += output_gradient_row[channel_idx] *
                                           nk_attention_plane_mxfp6e2m3_serial_(value_row, value_exponent, depth,
                                                                                channel_idx);
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp6e2m3_serial_(query_row, query_row_scales, key_row, key_exponent, depth,
                                                             scale2) -
                        log_sum_exp2);
                    nk_f32_t const score_gradient = weight *
                                                    (nk_scale_f32_serial_(weight_gradient, value_base) - row_dot) *
                                                    scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient *
                                                           nk_attention_plane_mxfp6e2m3_serial_(key_row, key_exponent,
                                                                                                depth, channel_idx);
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_mxfp6e2m3_serial_(
                                                             query_row, query_row_scales, channel_idx, 0);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] = nk_scale_f32_serial_(query_gradient_row[channel_idx], key_base);
            }
        }
    }
}

/** @c nk_attention_backward_mxfp4_serial_ for MXFP6 E3M2. */
NUMKONG_INLINE void nk_attention_backward_mxfp6e3m2_serial_(
    nk_mxfp6e3m2_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    key_value_head_idx * position_count * row_bytes;
        nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
        nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
        nk_i8_t const value_exponent =
            plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
        nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent),
                       value_base = nk_attention_plane_base_serial_(value_exponent);
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_u8_t const *key_row = keys_plane + position_idx * row_bytes;
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    nk_f32_t weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        weight_gradient += output_gradient_row[channel_idx] *
                                           nk_attention_plane_mxfp6e3m2_serial_(value_row, value_exponent, depth,
                                                                                channel_idx);
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp6e3m2_serial_(query_row, query_row_scales, key_row, key_exponent, depth,
                                                             scale2) -
                        log_sum_exp2);
                    nk_f32_t const score_gradient = weight *
                                                    (nk_scale_f32_serial_(weight_gradient, value_base) - row_dot) *
                                                    scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient *
                                                           nk_attention_plane_mxfp6e3m2_serial_(key_row, key_exponent,
                                                                                                depth, channel_idx);
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_mxfp6e3m2_serial_(
                                                             query_row, query_row_scales, channel_idx, 0);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] = nk_scale_f32_serial_(query_gradient_row[channel_idx], key_base);
            }
        }
    }
}

/** @c nk_attention_backward_mxfp4_serial_ for MXFP8 E4M3. */
NUMKONG_INLINE void nk_attention_backward_mxfp8e4m3_serial_(
    nk_mxfp8e4m3_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    key_value_head_idx * position_count * row_bytes;
        nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
        nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
        nk_i8_t const value_exponent =
            plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
        nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent),
                       value_base = nk_attention_plane_base_serial_(value_exponent);
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_u8_t const *key_row = keys_plane + position_idx * row_bytes;
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    nk_f32_t weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        weight_gradient += output_gradient_row[channel_idx] *
                                           nk_attention_plane_mxfp8e4m3_serial_(value_row, value_exponent, depth,
                                                                                channel_idx);
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp8e4m3_serial_(query_row, query_row_scales, key_row, key_exponent, depth,
                                                             scale2) -
                        log_sum_exp2);
                    nk_f32_t const score_gradient = weight *
                                                    (nk_scale_f32_serial_(weight_gradient, value_base) - row_dot) *
                                                    scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient *
                                                           nk_attention_plane_mxfp8e4m3_serial_(key_row, key_exponent,
                                                                                                depth, channel_idx);
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_mxfp8e4m3_serial_(
                                                             query_row, query_row_scales, channel_idx, 0);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] = nk_scale_f32_serial_(query_gradient_row[channel_idx], key_base);
            }
        }
    }
}

/** @c nk_attention_backward_mxfp4_serial_ for MXFP8 E5M2. */
NUMKONG_INLINE void nk_attention_backward_mxfp8e5m2_serial_(
    nk_mxfp8e5m2_cref_t const *queries, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_size_t query_gradient_stride,
    nk_size_t key_value_gradient_stride, nk_f32_t scale, nk_diagonal_band_t band, nk_size_t tasks_begin,
    nk_size_t tasks_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, 1,
                                                                                 depth * sizeof(nk_f32_t), key_offsets,
                                                                                 key_lengths, segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 32, row_bytes = depth * sizeof(nk_f32_t);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const query_gradient_stride_floats = query_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const gradient_stride_floats = key_value_gradient_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;

    if (tasks_end > segment_count * key_value_head_count) tasks_end = segment_count * key_value_head_count;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = key_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        nk_i64_t const first_position = nk_attention_first_position_serial_(row_count, position_count);
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    key_value_head_idx * position_count * row_bytes;
        nk_u8_t const *values_plane = keys_plane + key_value_head_count * position_count * row_bytes;
        nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
        nk_i8_t const value_exponent =
            plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
        nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent),
                       value_base = nk_attention_plane_base_serial_(value_exponent);
        nk_size_t const first_gradient = key_offsets[segment_idx] * gradient_stride_floats + key_value_head_idx * depth;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++)
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                key_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
                value_gradient[first_gradient + position_idx * gradient_stride_floats + channel_idx] = 0;
            }

        nk_size_t const head_end = (key_value_head_idx + 1) * head_group_size;
        for (nk_size_t head_idx = key_value_head_idx * head_group_size; head_idx < head_end; head_idx++) {
            for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
                nk_size_t const token = query_offsets[segment_idx] + row_idx;
                nk_u8_t const *query_row = query_codes + token * query_stride + head_idx * depth;
                nk_u8_t const *query_row_scales = query_scales + token * query_scales_stride + head_idx * depth / 32;
                nk_size_t const first_output = token * output_stride_floats + head_idx * depth;
                nk_f32_t const *output_row = output + first_output;
                nk_f32_t const *output_gradient_row = output_gradient + first_output;
                nk_f32_t *query_gradient_row = query_gradient + token * query_gradient_stride_floats + head_idx * depth;
                nk_f32_t const log_sum_exp2 = log_sum_exp[token * head_count + head_idx] * NUMKONG_F32_LOG2E_;
                nk_f32_t row_dot = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                    row_dot += output_gradient_row[channel_idx] * output_row[channel_idx];
                    query_gradient_row[channel_idx] = 0;
                }
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                    nk_u8_t const *key_row = keys_plane + position_idx * row_bytes;
                    nk_u8_t const *value_row = values_plane + position_idx * row_bytes;
                    nk_f32_t weight_gradient = 0;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                        weight_gradient += output_gradient_row[channel_idx] *
                                           nk_attention_plane_mxfp8e5m2_serial_(value_row, value_exponent, depth,
                                                                                channel_idx);
                    nk_f32_t const weight = nk_f32_exp2_serial_(
                        nk_attention_score_mxfp8e5m2_serial_(query_row, query_row_scales, key_row, key_exponent, depth,
                                                             scale2) -
                        log_sum_exp2);
                    nk_f32_t const score_gradient = weight *
                                                    (nk_scale_f32_serial_(weight_gradient, value_base) - row_dot) *
                                                    scale;
                    nk_f32_t *key_gradient_row = key_gradient + first_gradient + position_idx * gradient_stride_floats;
                    nk_f32_t *value_gradient_row = value_gradient + first_gradient +
                                                   position_idx * gradient_stride_floats;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                        value_gradient_row[channel_idx] += weight * output_gradient_row[channel_idx];
                        query_gradient_row[channel_idx] += score_gradient *
                                                           nk_attention_plane_mxfp8e5m2_serial_(key_row, key_exponent,
                                                                                                depth, channel_idx);
                        key_gradient_row[channel_idx] += score_gradient *
                                                         nk_attention_load_mxfp8e5m2_serial_(
                                                             query_row, query_row_scales, channel_idx, 0);
                    }
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    query_gradient_row[channel_idx] = nk_scale_f32_serial_(query_gradient_row[channel_idx], key_base);
            }
        }
    }
}

#if NUMKONG_TARGET_SERIAL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_attention_packed_bf16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_bf16_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                     depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                     tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_bf16_serial_(queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient,
                                       key_gradient, value_gradient, head_count, key_value_head_count, depth,
                                       query_offsets, query_stride, output_stride, query_gradient_stride,
                                       key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_nvfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_nvfp4_serial_(queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient,
                                        key_gradient, value_gradient, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride, output_stride, query_gradient_stride,
                                        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_mxfp4_serial_(queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient,
                                        key_gradient, value_gradient, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride, output_stride, query_gradient_stride,
                                        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e2m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_mxfp6e2m3_serial_(
        queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient, key_gradient, value_gradient,
        head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride, query_gradient_stride,
        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e3m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_mxfp6e3m2_serial_(
        queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient, key_gradient, value_gradient,
        head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride, query_gradient_stride,
        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_mxfp8e4m3_serial_(
        queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient, key_gradient, value_gradient,
        head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride, query_gradient_stride,
        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e5m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_backward_mxfp8e5m2_serial_(
        queries, key_value_packed, output, output_gradient, log_sum_exp, query_gradient, key_gradient, value_gradient,
        head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride, query_gradient_stride,
        key_value_gradient_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_f16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_f16_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                    depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                    tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_e4m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                     depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                     tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_nvfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_nvfp4_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                      depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                      tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_mxfp4_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                      depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                      tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_mxfp6e2m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_mxfp6e3m2_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_mxfp8e4m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_mxfp8e5m2_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_attention_pack_size_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_serial(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                    nk_i8_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                 key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_serial_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_i8_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                   depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                   tasks_end);
    return nk_success_k;
}

/** RoPE, the NeoX split-half rotary position embedding. Each row, a token @c x_stride bytes long,
 *  holds @c head_count heads of an even @c depth channels. Every pair @c i rotates channel @c i
 *  against its split-half partner i + depth / 2 by the per-token angle from the
 *  @b [rows,depth/2] cosine and sine grids, where row @c r starts at r × depth / 2 and is shared
 *  across heads, exactly a complex multiply by (cos, sin). The whole head is written, so the output
 *  @c y, @c y_stride bytes per row, may alias @c x for in-place rotation, and the caller bakes
 *  position lookup and M-RoPE axis assignment into the grids. */
#define nk_define_attention_rope_(input_type, load_and_convert, convert_and_store)                                     \
    NUMKONG_API nk_status_t nk_attention_rope_##input_type##_serial(                                                   \
        nk_##input_type##_t const *x, nk_f32_t const *cos, nk_f32_t const *sin, nk_##input_type##_t *y,                \
        nk_size_t rows, nk_size_t head_count, nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,                 \
        nk_stream_t stream) {                                                                                          \
        nk_assert_(stream == NUMKONG_NULL);                                                                            \
        nk_assert_(depth % 2 == 0);                                                                                    \
        nk_size_t const half_depth = depth / 2;                                                                        \
        for (nk_size_t r = 0; r != rows; ++r) {                                                                        \
            nk_f32_t const *cos_row = cos + r * half_depth;                                                            \
            nk_f32_t const *sin_row = sin + r * half_depth;                                                            \
            nk_##input_type##_t const *x_row = (nk_##input_type##_t const *)((unsigned char const *)x + r * x_stride); \
            nk_##input_type##_t *y_row = (nk_##input_type##_t *)((unsigned char *)y + r * y_stride);                   \
            for (nk_size_t h = 0; h != head_count; ++h) {                                                              \
                nk_##input_type##_t const *x_base = x_row + h * depth;                                                 \
                nk_##input_type##_t *y_base = y_row + h * depth;                                                       \
                for (nk_size_t i = 0; i != half_depth; ++i) {                                                          \
                    nk_f32_t low, high;                                                                                \
                    load_and_convert(x_base + i, &low);                                                                \
                    load_and_convert(x_base + i + half_depth, &high);                                                  \
                    nk_f32_t cosine = cos_row[i], sine = sin_row[i];                                                   \
                    nk_f32_t rotated_low = low * cosine - high * sine;                                                 \
                    nk_f32_t rotated_high = low * sine + high * cosine;                                                \
                    convert_and_store(&rotated_low, y_base + i);                                                       \
                    convert_and_store(&rotated_high, y_base + i + half_depth);                                         \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return nk_success_k;                                                                                           \
    }

nk_define_attention_rope_(e4m3, nk_e4m3_to_f32_serial_, nk_f32_to_e4m3_serial_)
nk_define_attention_rope_(bf16, nk_bf16_to_f32_serial_, nk_f32_to_bf16_serial_)
nk_define_attention_rope_(f32, nk_assign_from_to_, nk_assign_from_to_)
#undef nk_define_attention_rope_

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NUMKONG_TARGET_SERIAL

#if defined(__GNUC__) && !defined(__clang__) && NUMKONG_ARCH_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ATTENTION_SERIAL_H
