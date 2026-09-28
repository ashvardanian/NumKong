/**
 *  @file include/numkong/maxsim/genoa.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *  @brief SIMD-accelerated MaxSim, ColBERT late-interaction, for Genoa — bf16 only.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-512 VNNI, VPDPBUSD, for coarse i8 screening via icelake.h, and VDPBF16PS for bf16
 *  refinement. f32/f16 MaxSim variants live in icelake.h — this file only handles bf16 packing and
 *  its compute step.
 *
 *  @verbatim
 *  Intrinsic         Instruction  Genoa
 *  _mm512_dpbf16_ps  VDPBF16PS    6cy @ p01
 *  @endverbatim
 */
#ifndef NUMKONG_MAXSIM_GENOA_H
#define NUMKONG_MAXSIM_GENOA_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_GENOA

#include "numkong/types.h"
#include "numkong/maxsim/icelake.h" // `nk_maxsim_coarse_dots_icelake_`
#include "numkong/dot/genoa.h"      // `nk_dot_bf16_through_f32_genoa_`
#include "numkong/cast/icelake.h"   // `nk_e5m2x32_to_bf16x32_icelake_`
#include "numkong/dot/skylake.h"    // `nk_dot_through_f32_finalize_skylake_`
#include "numkong/maxsim/serial.h"  // `nk_maxsim_packed_header_setup_`, `nk_maxsim_pack_size_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`

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

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_genoa(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_bf16_t), 64);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_genoa(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_genoa( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 64, element_bytes,
                                                               nk_cap_genoa_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_genoa_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_through_f32_genoa_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_genoa( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_genoa_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_genoa_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_icelake_, nk_maxsim_refine_bf16_genoa_);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_GENOA
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MAXSIM_GENOA_H
