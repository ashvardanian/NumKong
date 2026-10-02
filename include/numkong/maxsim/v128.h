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
 */
#ifndef NUMKONG_MAXSIM_V128_H
#define NUMKONG_MAXSIM_V128_H

#if NUMKONG_ARCH_WASM_V128_

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // `nk_maxsim_packed_header_t`
#include "numkong/cast/serial.h"   // `nk_bf16_to_f32_`, `nk_f16_to_f32_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_bf16_v128_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_v128relaxed_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 63.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_f32_v128_( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_v128relaxed_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 63.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/** Quantizes and copies @p vectors into the layout that @c v128relaxed multiplies, recording that
 *  capability; its pack runs on SIMD128 alone. */
NUMKONG_INLINE void nk_maxsim_pack_f16_v128_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {
    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_v128relaxed_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 63.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_WASM_V128_
#endif // NUMKONG_MAXSIM_V128_H
