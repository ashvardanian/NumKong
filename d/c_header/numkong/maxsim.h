/**
 *  @brief SIMD-accelerated MaxSim (ColBERT Late Interaction).
 *  @file include/numkong/maxsim.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  Computes angular distance late-interaction: result = Σᵢ minⱼ angular(qᵢ, dⱼ).
 *  Angular distance = 1 - dot(q, d) / sqrt(||q||² × ||d||²), clamped >= 0.
 *
 *  Strategy: coarse i8-quantized screening with running argmax (dot as proxy for argmin angular),
 *  then full-precision refinement of the winning pairs via nk_dot_* primitives,
 *  finalized with angular distance and accumulated with `f64`.
 *
 *  Precision policy:
 *  - `f32` inputs keep packed payloads and metadata narrow for memory bandwidth.
 *  - The refined scores and final late-interaction sum widen to `f64`.
 *
 *  It implements several operations:
 *
 *  - "maxsim_packed" - computing MaxSim where both Q and D are pre-packed into optimal form
 *  - "maxsim_packed_size" - estimating the memory requirements for external malloc
 *  - "maxsim_pack" - performing the pre-processing (quantization + original copy)
 *
 *  @section maxsim_api Two-Phase API
 *
 *  @code{.c}
 *  // Pack query and document matrices
 *  nk_size_t query_bytes = nk_maxsim_packed_size_bf16(query_count, depth);
 *  nk_size_t document_bytes = nk_maxsim_packed_size_bf16(document_count, depth);
 *  void *query_packed = malloc(query_bytes);
 *  void *document_packed = malloc(document_bytes);
 *  nk_maxsim_pack_bf16(queries, query_count, depth, depth * sizeof(nk_bf16_t), query_packed);
 *  nk_maxsim_pack_bf16(documents, document_count, depth, depth * sizeof(nk_bf16_t), document_packed);
 *
 *  // Compute MaxSim score
 *  nk_f32_t score;
 *  nk_maxsim_packed_bf16(query_packed, document_packed, query_count, document_count, depth, &score);
 *  @endcode
 *
 *  @section maxsim_packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  The packed format is backend-specific: different ISAs use different i8 depth padding
 *  and clamp ranges. Pack with the matching ISA's pack function.
 *
 *  @section maxsim_isa_support ISA Support
 *
 *  Currently implemented:
 *  - Serial: scalar reference (all platforms)
 *  - Haswell: AVX2 VPMADDUBSW coarse [-79,79] + bias correction (bf16/f32/f16)
 *  - Icelake: AVX-512 VNNI VPDPBUSD coarse (f32/f16)
 *  - Genoa: AVX-512 VNNI coarse + VDPBF16PS refinement (bf16 only)
 *  - NEONSDOT: ARM SDOT (vdotq_s32) coarse, no bias correction (bf16/f32/f16)
 *  - SME: ARM fused BFMOPA (existing, unchanged)
 */
#ifndef NK_MAXSIM_H
#define NK_MAXSIM_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for a maxsim vector set.
 *  @param[in] vector_count The number of vectors to pack.
 *  @param[in] depth The number of dimensions per vector.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_bf16(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_f32(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_maxsim_packed_size_f16(nk_size_t vector_count, nk_size_t depth);

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors The input vectors in row-major order.
 *  @param[in] vector_count The number of vectors.
 *  @param[in] depth The number of dimensions per vector.
 *  @param[in] stride The row stride in bytes for the input vectors.
 *  @param[out] packed The output packed buffer from nk_maxsim_packed_size_bf16.
 */
NK_DYNAMIC void nk_maxsim_pack_bf16(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                    void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_DYNAMIC void nk_maxsim_pack_f32(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                   void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_DYNAMIC void nk_maxsim_pack_f16(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                   void *packed);

/**
 *  @brief Computes angular distance late-interaction on pre-packed vectors.
 *  Returns Σᵢ minⱼ angular(qᵢ, dⱼ) where angular = 1 - dot / sqrt(||q||² × ||d||²).
 *
 *  @param[in] query_packed Packed query vectors (from nk_maxsim_pack_bf16).
 *  @param[in] document_packed Packed document vectors (from nk_maxsim_pack_bf16).
 *  @param[in] query_count Number of query vectors.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[out] result Pointer to store the sum of per-query minimum angular distances.
 */
NK_DYNAMIC void nk_maxsim_packed_bf16(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                      nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_DYNAMIC void nk_maxsim_packed_f32(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                     nk_size_t document_count, nk_size_t depth, nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_DYNAMIC void nk_maxsim_packed_f16(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                     nk_size_t document_count, nk_size_t depth, nk_f32_t *result);

/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_serial(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_serial(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_serial(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_serial(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                          nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_serial(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                         nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_serial(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                         nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_serial(void const *query_packed, void const *document_packed,
                                            nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                            nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_serial(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                           nk_size_t document_count, nk_size_t depth, nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_serial(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                           nk_size_t document_count, nk_size_t depth, nk_f32_t *result);

#if NK_TARGET_ICELAKE
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_icelake(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_icelake(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_icelake(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                          nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_icelake(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                          nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_icelake(void const *query_packed, void const *document_packed,
                                            nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                            nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_icelake(void const *query_packed, void const *document_packed,
                                            nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                            nk_f32_t *result);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_genoa(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                         nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_genoa(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                           nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_sapphireamx(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_sapphireamx(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_sapphireamx(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                               nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_sapphireamx(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                              nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_sapphireamx(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                              nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_sapphireamx(void const *query_packed, void const *document_packed,
                                                 nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                 nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_sapphireamx(void const *query_packed, void const *document_packed,
                                                nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_sapphireamx(void const *query_packed, void const *document_packed,
                                                nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                nk_f32_t *result);
#endif // NK_TARGET_SAPPHIREAMX

#if NK_TARGET_HASWELL
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_haswell(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_haswell(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_haswell(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                           nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_haswell(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                          nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_haswell(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                          nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_haswell(void const *query_packed, void const *document_packed,
                                             nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                             nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_haswell(void const *query_packed, void const *document_packed,
                                            nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                            nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_haswell(void const *query_packed, void const *document_packed,
                                            nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                            nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_ALDER
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_alder(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_alder(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_alder(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_alder(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                         nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_alder(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                        nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_alder(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                        nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_alder(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                           nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_alder(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                          nk_size_t document_count, nk_size_t depth, nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_alder(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                          nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
#endif // NK_TARGET_ALDER

#if NK_TARGET_V128RELAXED
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_v128relaxed(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_v128relaxed(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_v128relaxed(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                               nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                              nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                              nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_v128relaxed(void const *query_packed, void const *document_packed,
                                                 nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                 nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_v128relaxed(void const *query_packed, void const *document_packed,
                                                nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_v128relaxed(void const *query_packed, void const *document_packed,
                                                nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                nk_f32_t *result);
#endif // NK_TARGET_V128RELAXED

#if NK_TARGET_NEONSDOT
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_neonsdot(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_neonsdot(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_neonsdot(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_neonsdot(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                            nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_neonsdot(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                           nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_neonsdot(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                           nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_neonsdot(void const *query_packed, void const *document_packed,
                                              nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                              nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_neonsdot(void const *query_packed, void const *document_packed,
                                             nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                             nk_f64_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_neonsdot(void const *query_packed, void const *document_packed,
                                             nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                             nk_f32_t *result);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_SME
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_sme(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_sme(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_sme(nk_size_t vector_count, nk_size_t depth);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_bf16_sme(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                       nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f16_sme(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                      nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_pack_bf16 */
NK_PUBLIC void nk_maxsim_pack_f32_sme(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                      nk_size_t stride, void *packed);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_bf16_sme(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                         nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f16_sme(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                        nk_size_t document_count, nk_size_t depth, nk_f32_t *result);
/** @copydoc nk_maxsim_packed_bf16 */
NK_PUBLIC void nk_maxsim_packed_f32_sme(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                        nk_size_t document_count, nk_size_t depth, nk_f64_t *result);
#endif // NK_TARGET_SME

/**
 *  @brief Returns the output dtype for MaxSim late-interaction.
 */
NK_INTERNAL nk_dtype_t nk_maxsim_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/maxsim/serial.h"
#include "numkong/maxsim/haswell.h"
#include "numkong/maxsim/alder.h"
#include "numkong/maxsim/icelake.h"
#include "numkong/maxsim/genoa.h"
#include "numkong/maxsim/sapphireamx.h"
#include "numkong/maxsim/neonsdot.h"
#include "numkong/maxsim/sme.h"
#include "numkong/maxsim/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16(nk_size_t vector_count, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_bf16_sme(vector_count, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_maxsim_packed_size_bf16_sapphireamx(vector_count, depth);
#elif NK_TARGET_GENOA
    return nk_maxsim_packed_size_bf16_genoa(vector_count, depth);
#elif NK_TARGET_ALDER
    return nk_maxsim_packed_size_bf16_alder(vector_count, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_bf16_haswell(vector_count, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_bf16_neonsdot(vector_count, depth);
#elif NK_TARGET_V128RELAXED
    return nk_maxsim_packed_size_bf16_v128relaxed(vector_count, depth);
#else
    return nk_maxsim_packed_size_bf16_serial(vector_count, depth);
#endif
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32(nk_size_t vector_count, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_f32_sme(vector_count, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_maxsim_packed_size_f32_sapphireamx(vector_count, depth);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_size_f32_icelake(vector_count, depth);
#elif NK_TARGET_ALDER
    return nk_maxsim_packed_size_f32_alder(vector_count, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_f32_haswell(vector_count, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_f32_neonsdot(vector_count, depth);
#elif NK_TARGET_V128RELAXED
    return nk_maxsim_packed_size_f32_v128relaxed(vector_count, depth);
#else
    return nk_maxsim_packed_size_f32_serial(vector_count, depth);
#endif
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16(nk_size_t vector_count, nk_size_t depth) {
#if NK_TARGET_SME
    return nk_maxsim_packed_size_f16_sme(vector_count, depth);
#elif NK_TARGET_SAPPHIREAMX
    return nk_maxsim_packed_size_f16_sapphireamx(vector_count, depth);
#elif NK_TARGET_ICELAKE
    return nk_maxsim_packed_size_f16_icelake(vector_count, depth);
#elif NK_TARGET_ALDER
    return nk_maxsim_packed_size_f16_alder(vector_count, depth);
#elif NK_TARGET_HASWELL
    return nk_maxsim_packed_size_f16_haswell(vector_count, depth);
#elif NK_TARGET_NEONSDOT
    return nk_maxsim_packed_size_f16_neonsdot(vector_count, depth);
#elif NK_TARGET_V128RELAXED
    return nk_maxsim_packed_size_f16_v128relaxed(vector_count, depth);
#else
    return nk_maxsim_packed_size_f16_serial(vector_count, depth);
#endif
}

NK_PUBLIC void nk_maxsim_pack_bf16(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                   void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_bf16_sme(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_pack_bf16_sapphireamx(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_GENOA
    nk_maxsim_pack_bf16_genoa(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_ALDER
    nk_maxsim_pack_bf16_alder(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_bf16_haswell(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_bf16_neonsdot(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_pack_bf16_v128relaxed(vectors, vector_count, depth, stride, packed);
#else
    nk_maxsim_pack_bf16_serial(vectors, vector_count, depth, stride, packed);
#endif
}

NK_PUBLIC void nk_maxsim_pack_f32(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                  void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_f32_sme(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_pack_f32_sapphireamx(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_ICELAKE
    nk_maxsim_pack_f32_icelake(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_ALDER
    nk_maxsim_pack_f32_alder(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_f32_haswell(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_f32_neonsdot(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_pack_f32_v128relaxed(vectors, vector_count, depth, stride, packed);
#else
    nk_maxsim_pack_f32_serial(vectors, vector_count, depth, stride, packed);
#endif
}

NK_PUBLIC void nk_maxsim_pack_f16(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                  void *packed) {
#if NK_TARGET_SME
    nk_maxsim_pack_f16_sme(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_pack_f16_sapphireamx(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_ICELAKE
    nk_maxsim_pack_f16_icelake(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_ALDER
    nk_maxsim_pack_f16_alder(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_HASWELL
    nk_maxsim_pack_f16_haswell(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_pack_f16_neonsdot(vectors, vector_count, depth, stride, packed);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_pack_f16_v128relaxed(vectors, vector_count, depth, stride, packed);
#else
    nk_maxsim_pack_f16_serial(vectors, vector_count, depth, stride, packed);
#endif
}

NK_PUBLIC void nk_maxsim_packed_bf16(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                     nk_size_t document_count, nk_size_t depth, nk_f32_t *result) {
#if NK_TARGET_SME
    nk_maxsim_packed_bf16_sme(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_packed_bf16_sapphireamx(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_GENOA
    nk_maxsim_packed_bf16_genoa(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_ALDER
    nk_maxsim_packed_bf16_alder(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_HASWELL
    nk_maxsim_packed_bf16_haswell(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_packed_bf16_neonsdot(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_packed_bf16_v128relaxed(query_packed, document_packed, query_count, document_count, depth, result);
#else
    nk_maxsim_packed_bf16_serial(query_packed, document_packed, query_count, document_count, depth, result);
#endif
}

NK_PUBLIC void nk_maxsim_packed_f32(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                    nk_size_t document_count, nk_size_t depth, nk_f64_t *result) {
#if NK_TARGET_SME
    nk_maxsim_packed_f32_sme(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_packed_f32_sapphireamx(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_ICELAKE
    nk_maxsim_packed_f32_icelake(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_ALDER
    nk_maxsim_packed_f32_alder(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_HASWELL
    nk_maxsim_packed_f32_haswell(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_packed_f32_neonsdot(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_packed_f32_v128relaxed(query_packed, document_packed, query_count, document_count, depth, result);
#else
    nk_maxsim_packed_f32_serial(query_packed, document_packed, query_count, document_count, depth, result);
#endif
}

NK_PUBLIC void nk_maxsim_packed_f16(void const *query_packed, void const *document_packed, nk_size_t query_count,
                                    nk_size_t document_count, nk_size_t depth, nk_f32_t *result) {
#if NK_TARGET_SME
    nk_maxsim_packed_f16_sme(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_SAPPHIREAMX
    nk_maxsim_packed_f16_sapphireamx(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_ICELAKE
    nk_maxsim_packed_f16_icelake(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_ALDER
    nk_maxsim_packed_f16_alder(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_HASWELL
    nk_maxsim_packed_f16_haswell(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_NEONSDOT
    nk_maxsim_packed_f16_neonsdot(query_packed, document_packed, query_count, document_count, depth, result);
#elif NK_TARGET_V128RELAXED
    nk_maxsim_packed_f16_v128relaxed(query_packed, document_packed, query_count, document_count, depth, result);
#else
    nk_maxsim_packed_f16_serial(query_packed, document_packed, query_count, document_count, depth, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MAXSIM_H
