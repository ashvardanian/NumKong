/**
 *  @file include/numkong/maxsim/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, packing for WASM SIMD128.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Packs vectors into the i8 coarse-screening layout of `maxsim/v128relaxed.h`, the only capability
 *  that runs these helpers, as only it multiplies: quantization keeps both operands within the i7
 *  range [-63, 63].
 *
 *  Each pack reads its source once: the first pass loads a row, stores it into the packed originals
 *  and accumulates its abs-max and F64 sum of squares. The second pass quantizes the originals row
 *  it just wrote.
 */
#ifndef NUMKONG_MAXSIM_V128_H
#define NUMKONG_MAXSIM_V128_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_ARCH_WASM_V128_

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // `nk_maxsim_packed_header_t`
#include "numkong/cast/serial.h"   // `nk_partial_load_b16x8_serial_`
#include "numkong/cast/v128.h"     // `nk_f16x4_to_f32x4_v128_`
#include "numkong/reduce/v128.h"   // `nk_reduce_max_f32x4_v128_`, `nk_reduce_add_f64x2_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/** Raises @p absmax_f32x4 to the magnitudes of @p values_f32x4 and adds their squares, each exact
 *  in F64, to @p sumsq_f64x2. */
NUMKONG_INLINE void nk_maxsim_moments_f32x4_v128_(v128_t values_f32x4, v128_t *absmax_f32x4, v128_t *sumsq_f64x2) {
    // Pseudo-maximum keeps the running lane when the new one is NaN, as the serial comparison does
    *absmax_f32x4 = wasm_f32x4_pmax(*absmax_f32x4, wasm_f32x4_abs(values_f32x4));
    v128_t low_f64x2 = wasm_f64x2_promote_low_f32x4(values_f32x4);
    v128_t high_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(values_f32x4, values_f32x4, 2, 3, 0, 1));
    *sumsq_f64x2 = wasm_f64x2_add(*sumsq_f64x2, wasm_f64x2_mul(low_f64x2, low_f64x2));
    *sumsq_f64x2 = wasm_f64x2_add(*sumsq_f64x2, wasm_f64x2_mul(high_f64x2, high_f64x2));
}

/** Divides @p values_f32x4 by @p scale_f32x4, rounds half away from zero, clamps to [-63, 63]. */
NUMKONG_INLINE v128_t nk_maxsim_quantize_f32x4_v128_(v128_t values_f32x4, v128_t scale_f32x4) {
    v128_t scaled_f32x4 = wasm_f32x4_div(values_f32x4, scale_f32x4);
    v128_t half_f32x4 = wasm_v128_or(wasm_v128_and(scaled_f32x4, wasm_i32x4_splat((int)0x80000000)),
                                     wasm_f32x4_splat(0.5f));
    v128_t codes_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_add(scaled_f32x4, half_f32x4));
    return wasm_i32x4_max(wasm_i32x4_min(codes_i32x4, wasm_i32x4_splat(63)), wasm_i32x4_splat(-63));
}

/** Quantizes 16 values by @p scale_f32x4, stores their codes to @p codes and returns @p sum_i32x4
 *  plus the codes. */
NUMKONG_INLINE v128_t nk_maxsim_quantize_f32x16_v128_(v128_t first_f32x4, v128_t second_f32x4, v128_t third_f32x4,
                                                      v128_t fourth_f32x4, v128_t scale_f32x4, nk_i8_t *codes,
                                                      v128_t sum_i32x4) {
    v128_t low_i16x8 = wasm_i16x8_narrow_i32x4(nk_maxsim_quantize_f32x4_v128_(first_f32x4, scale_f32x4),
                                               nk_maxsim_quantize_f32x4_v128_(second_f32x4, scale_f32x4));
    v128_t high_i16x8 = wasm_i16x8_narrow_i32x4(nk_maxsim_quantize_f32x4_v128_(third_f32x4, scale_f32x4),
                                                nk_maxsim_quantize_f32x4_v128_(fourth_f32x4, scale_f32x4));
    v128_t codes_i8x16 = wasm_i8x16_narrow_i16x8(low_i16x8, high_i16x8);
    wasm_v128_store(codes, codes_i8x16);
    return wasm_i32x4_add(sum_i32x4, wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_extadd_pairwise_i8x16(codes_i8x16)));
}

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_bf16_v128_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_bf16_t),
                                                                     nk_cap_v128relaxed_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    v128_t const high_mask_u32x4 = wasm_i32x4_splat((int)0xFFFF0000);

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_bf16_t *original_row = (nk_bf16_t *)(originals + vector_index * header->original_stride);
        nk_i8_t *quantized_row = quantized_i8 + vector_index * depth_i8_padded;

        v128_t absmax_f32x4 = wasm_f32x4_splat(0), sumsq_f64x2 = wasm_f64x2_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 8) {
            nk_size_t const count = nk_min_of_two(depth - depth_index, 8);
            nk_b128_vec_t raw_vec;
            if (count == 8) nk_load_b128_v128_(source_row + depth_index * sizeof(nk_bf16_t), &raw_vec);
            else nk_partial_load_b16x8_serial_(source_row + depth_index * sizeof(nk_bf16_t), &raw_vec, count);
            nk_store_b128_v128_(&raw_vec, original_row + depth_index);
            nk_maxsim_moments_f32x4_v128_(wasm_i32x4_shl(raw_vec.v128, 16), &absmax_f32x4, &sumsq_f64x2);
            nk_maxsim_moments_f32x4_v128_(wasm_v128_and(raw_vec.v128, high_mask_u32x4), &absmax_f32x4, &sumsq_f64x2);
        }
        nk_f32_t scale = nk_reduce_max_f32x4_v128_(absmax_f32x4) / 63.0f;
        if (scale == 0.0f) scale = 1.0f;
        nk_f64_t const norm_squared = nk_reduce_add_f64x2_v128_(sumsq_f64x2);

        v128_t const scale_f32x4 = wasm_f32x4_splat(scale);
        v128_t sum_i32x4 = wasm_i32x4_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 16) {
            v128_t first_u16x8 = wasm_v128_load(original_row + depth_index);
            v128_t second_u16x8 = wasm_v128_load(original_row + depth_index + 8);
            sum_i32x4 = nk_maxsim_quantize_f32x16_v128_(wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(first_u16x8), 16),
                                                        wasm_i32x4_shl(wasm_u32x4_extend_high_u16x8(first_u16x8), 16),
                                                        wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(second_u16x8), 16),
                                                        wasm_i32x4_shl(wasm_u32x4_extend_high_u16x8(second_u16x8), 16),
                                                        scale_f32x4, quantized_row + depth_index, sum_i32x4);
        }
        metadata[vector_index].inverse_norm_f64 = norm_squared > 0.0 ? nk_f64_rsqrt_(norm_squared) : 0.0;
        metadata[vector_index].screen_weight_f32 = scale * (nk_f32_t)metadata[vector_index].inverse_norm_f64;
        metadata[vector_index].sum_i8_i32 = nk_reduce_add_i32x4_v128_(sum_i32x4);
    }
}

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_f32_v128_( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_f32_t),
                                                                     nk_cap_v128relaxed_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t *original_row = (nk_f32_t *)(originals + vector_index * header->original_stride);
        nk_i8_t *quantized_row = quantized_i8 + vector_index * depth_i8_padded;

        v128_t absmax_f32x4 = wasm_f32x4_splat(0), sumsq_f64x2 = wasm_f64x2_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 4) {
            nk_size_t const count = nk_min_of_two(depth - depth_index, 4);
            nk_b128_vec_t raw_vec;
            if (count == 4) nk_load_b128_v128_(source_row + depth_index * sizeof(nk_f32_t), &raw_vec);
            else nk_partial_load_b32x4_serial_(source_row + depth_index * sizeof(nk_f32_t), &raw_vec, count);
            nk_store_b128_v128_(&raw_vec, original_row + depth_index);
            nk_maxsim_moments_f32x4_v128_(raw_vec.v128, &absmax_f32x4, &sumsq_f64x2);
        }
        nk_f32_t scale = nk_reduce_max_f32x4_v128_(absmax_f32x4) / 63.0f;
        if (scale == 0.0f) scale = 1.0f;
        nk_f64_t const norm_squared = nk_reduce_add_f64x2_v128_(sumsq_f64x2);

        v128_t const scale_f32x4 = wasm_f32x4_splat(scale);
        v128_t sum_i32x4 = wasm_i32x4_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 16)
            sum_i32x4 = nk_maxsim_quantize_f32x16_v128_(
                wasm_v128_load(original_row + depth_index), wasm_v128_load(original_row + depth_index + 4),
                wasm_v128_load(original_row + depth_index + 8), wasm_v128_load(original_row + depth_index + 12),
                scale_f32x4, quantized_row + depth_index, sum_i32x4);
        metadata[vector_index].inverse_norm_f64 = norm_squared > 0.0 ? nk_f64_rsqrt_(norm_squared) : 0.0;
        metadata[vector_index].screen_weight_f32 = scale * (nk_f32_t)metadata[vector_index].inverse_norm_f64;
        metadata[vector_index].sum_i8_i32 = nk_reduce_add_i32x4_v128_(sum_i32x4);
    }
}

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_f16_v128_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_f16_t),
                                                                     nk_cap_v128relaxed_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f16_t *original_row = (nk_f16_t *)(originals + vector_index * header->original_stride);
        nk_i8_t *quantized_row = quantized_i8 + vector_index * depth_i8_padded;

        v128_t absmax_f32x4 = wasm_f32x4_splat(0), sumsq_f64x2 = wasm_f64x2_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 8) {
            nk_size_t const count = nk_min_of_two(depth - depth_index, 8);
            nk_b128_vec_t raw_vec;
            if (count == 8) nk_load_b128_v128_(source_row + depth_index * sizeof(nk_f16_t), &raw_vec);
            else nk_partial_load_b16x8_serial_(source_row + depth_index * sizeof(nk_f16_t), &raw_vec, count);
            nk_store_b128_v128_(&raw_vec, original_row + depth_index);
            nk_b64_vec_t low_vec, high_vec;
            low_vec.u64 = raw_vec.u64s[0], high_vec.u64 = raw_vec.u64s[1];
            nk_maxsim_moments_f32x4_v128_(nk_f16x4_to_f32x4_v128_(low_vec).v128, &absmax_f32x4, &sumsq_f64x2);
            nk_maxsim_moments_f32x4_v128_(nk_f16x4_to_f32x4_v128_(high_vec).v128, &absmax_f32x4, &sumsq_f64x2);
        }
        nk_f32_t scale = nk_reduce_max_f32x4_v128_(absmax_f32x4) / 63.0f;
        if (scale == 0.0f) scale = 1.0f;
        nk_f64_t const norm_squared = nk_reduce_add_f64x2_v128_(sumsq_f64x2);

        v128_t const scale_f32x4 = wasm_f32x4_splat(scale);
        v128_t sum_i32x4 = wasm_i32x4_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth; depth_index += 16) {
            nk_b256_vec_t raw_vec;
            nk_load_b256_v128_(original_row + depth_index, &raw_vec);
            nk_b64_vec_t first_vec, second_vec, third_vec, fourth_vec;
            first_vec.u64 = raw_vec.u64s[0], second_vec.u64 = raw_vec.u64s[1];
            third_vec.u64 = raw_vec.u64s[2], fourth_vec.u64 = raw_vec.u64s[3];
            sum_i32x4 = nk_maxsim_quantize_f32x16_v128_(
                nk_f16x4_to_f32x4_v128_(first_vec).v128, nk_f16x4_to_f32x4_v128_(second_vec).v128,
                nk_f16x4_to_f32x4_v128_(third_vec).v128, nk_f16x4_to_f32x4_v128_(fourth_vec).v128, scale_f32x4,
                quantized_row + depth_index, sum_i32x4);
        }
        metadata[vector_index].inverse_norm_f64 = norm_squared > 0.0 ? nk_f64_rsqrt_(norm_squared) : 0.0;
        metadata[vector_index].screen_weight_f32 = scale * (nk_f32_t)metadata[vector_index].inverse_norm_f64;
        metadata[vector_index].sum_i8_i32 = nk_reduce_add_i32x4_v128_(sum_i32x4);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_WASM_V128_
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_MAXSIM_V128_H
