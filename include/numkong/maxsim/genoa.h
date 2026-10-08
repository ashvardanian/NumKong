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
#include "numkong/maxsim/serial.h"  // `nk_maxsim_packed_header_setup_serial_`, `nk_maxsim_pack_size_serial_`
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
    *bytes = nk_maxsim_pack_size_serial_(vector_count, depth, sizeof(nk_bf16_t), 64);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_genoa(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_serial_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_genoa( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_serial_(packed, vector_count, depth, 64, element_bytes,
                                                                      nk_cap_genoa_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_bf16_icelake_((nk_bf16_t const *)((char const *)vectors + vector_index * stride), depth,
                                            (nk_bf16_t *)(originals + vector_index * original_stride),
                                            quantized_i8 + vector_index * depth_i8_padded, 64, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_genoa_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_through_f32_genoa_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_genoa( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(query_packed, nk_cap_genoa_k) ||
        !nk_maxsim_packed_by_serial_(document_packed, nk_cap_genoa_k))
        return nk_pack_mismatch_k;

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_serial_(query_packed, document_packed);
    nk_f64_t const poison = nk_maxsim_packed_poison_serial_(&regions, query_count, document_count);
    if (poison != 0) {
        *result = (nk_f32_t)poison;
        return nk_success_k;
    }
    nk_size_t const weights_stride = sizeof(nk_maxsim_vector_metadata_t) / sizeof(nk_f32_t);
    nk_f32_t const residue = 0.5f * nk_sqrt_f32_serial_((nk_f32_t)depth);
    nk_i32_t dots[32 * 128];
    nk_u32_t candidates[128];
    nk_f64_t total_angular_distance = 0.0, total_compensation = 0.0;

    for (nk_size_t query_start = 0; query_start < query_count; query_start += 32) {
        nk_size_t const query_tile = query_count - query_start < 32 ? query_count - query_start : 32;
        nk_maxsim_screen_error_t errors[32];
        nk_f32_t lower_bounds[32];
        nk_f64_t best_cosines[32];
        nk_maxsim_query_tile_begin_serial_(&regions, query_start, query_tile, residue, depth, errors, lower_bounds,
                                           best_cosines);

        for (nk_size_t document_start = 0; document_start < document_count; document_start += 128) {
            nk_size_t const document_tile = document_count - document_start < 128 ? document_count - document_start
                                                                                  : 128;
            nk_maxsim_coarse_dots_icelake_(regions.query_quantized + query_start * regions.depth_i8_padded,
                                           regions.document_quantized + document_start * regions.depth_i8_padded,
                                           regions.document_metadata + document_start, query_tile, document_tile,
                                           regions.depth_i8_padded, dots);
            nk_f32_t const *weights = &regions.document_metadata[document_start].screen_weight_f32;

            for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
                nk_size_t const query_global_index = query_start + query_index;
                nk_size_t const candidate_count = nk_maxsim_screen_query_serial_(
                    dots + query_index * document_tile, weights, weights_stride, document_tile, errors[query_index],
                    &lower_bounds[query_index], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = document_start + candidates[candidate_index];
                    nk_f64_t const dot = nk_maxsim_refine_bf16_genoa_(
                        regions.query_originals + query_global_index * regions.query_original_stride,
                        regions.document_originals + document_index * regions.document_original_stride, depth);
                    nk_maxsim_best_cosine_update_serial_(
                        dot, regions.query_metadata[query_global_index].inverse_norm_f64,
                        regions.document_metadata[document_index].inverse_norm_f64, &best_cosines[query_index]);
                }
            }
        }

        nk_maxsim_angular_accumulate_serial_(best_cosines, query_tile, &total_angular_distance, &total_compensation);
    }
    *result = (nk_f32_t)(total_angular_distance + total_compensation);
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
