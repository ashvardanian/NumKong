/**
 *  @brief SIMD-accelerated MaxSim (ColBERT late-interaction) for Genoa — bf16 only.
 *  @file include/numkong/maxsim/genoa.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-512 VNNI (VPDPBUSD) for coarse i8 screening via icelake.h, and VDPBF16PS for bf16 refinement.
 *  f32/f16 MaxSim variants live in icelake.h — this file only provides bf16 pack and compute.
 *
 *      Intrinsic                   Instruction     Genoa (Zen4)
 *      _mm512_dpbf16_ps            VDPBF16PS       6cy @ p01 (512-bit)
 */
#ifndef NK_MAXSIM_GENOA_H
#define NK_MAXSIM_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/maxsim/icelake.h" // `nk_maxsim_coarse_argmax_icelake_`
#include "numkong/dot.h"            // `nk_dot_bf16`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                   \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512bf16", "f16c", "fma", \
                   "bmi", "bmi2")
#endif

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_genoa(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_bf16_t), 64);
}

NK_PUBLIC void nk_maxsim_pack_bf16_genoa( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 64, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_serial,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_packed_bf16_genoa( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_icelake_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                         regions.document_quantized, regions.document_metadata, chunk_size,
                                         document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_bf16((nk_bf16_t const *)(regions.query_originals +
                                            (chunk_start + query_index) * regions.query_original_stride),
                        (nk_bf16_t const *)(regions.document_originals +
                                            best_document_index * regions.document_original_stride),
                        depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_
#endif // NK_MAXSIM_GENOA_H
