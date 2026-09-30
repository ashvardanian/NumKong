/**
 *  @file include/numkong/maxsim.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *  @brief SIMD-accelerated MaxSim, ColBERT late interaction.
 *
 *  Computes angular distance late-interaction: result = Σᵢ minⱼ angular(qᵢ, dⱼ).
 *  Angular distance = 1 - dot(q, d) / sqrt(||q||² × ||d||²), clamped >= 0.
 *
 *  Strategy: coarse i8-quantized screening with a running argmax over i8 dots, each weighted by its
 *  document's scale / ‖d‖ to rank by cosine, then full-precision refinement of the winning pairs
 *  via @c nk_dot_* primitives, finalized with angular distance and accumulated with @c f64.
 *
 *  Precision policy:
 *  - @c f32 inputs keep packed payloads and metadata narrow for memory bandwidth.
 *  - The refined scores and final late-interaction sum widen to @c f64.
 *
 *  It implements several operations:
 *
 *  - @c maxsim_packed - computing MaxSim where both Q and D are pre-packed into optimal form
 *  - @c maxsim_pack_size - estimating the memory requirements for external malloc
 *  - @c maxsim_pack - performing the pre-processing, quantization plus original copy
 *
 *  @section maxsim_api Two-Phase API
 *
 *  @code{.c}
 *  // Pack query and document matrices with the same capabilities, as the packs record theirs
 *  nk_capability_t caps = nk_cap_serial_k;
 *  nk_cpu_capabilities_enabled(&caps);
 *  nk_size_t query_bytes = 0, document_bytes = 0;
 *  nk_maxsim_pack_size_bf16_best(query_count, depth, caps, &query_bytes);
 *  nk_maxsim_pack_size_bf16_best(document_count, depth, caps, &document_bytes);
 *  void *query_packed = malloc(query_bytes);
 *  void *document_packed = malloc(document_bytes);
 *  nk_maxsim_pack_bf16_best(queries, query_count, depth, depth * sizeof(nk_bf16_t), query_packed, caps, NULL);
 *  nk_maxsim_pack_bf16_best(documents, document_count, depth, depth * sizeof(nk_bf16_t), document_packed, caps, NULL);
 *
 *  // Compute MaxSim score
 *  nk_f32_t score;
 *  nk_maxsim_packed_bf16_best(query_packed, document_packed, query_count, document_count, depth, &score, caps, NULL);
 *  @endcode
 *
 *  @section maxsim_packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors 64B-aligned] [metadata 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  The packed format is backend-specific: different ISAs use different i8 depth padding and clamp
 *  ranges. Pack with the matching ISA's pack function.
 *
 *  @section maxsim_isa_support ISA Support
 *
 *  Currently implemented:
 *  - Serial: scalar reference, all platforms
 *  - Haswell: AVX2 VPMADDUBSW coarse [-79,79] plus bias correction, bf16/f32/f16
 *  - Icelake: AVX-512 VNNI VPDPBUSD coarse, f32/f16
 *  - Genoa: AVX-512 VNNI coarse plus VDPBF16PS refinement, bf16 only
 *  - NEONSDOT: ARM SDOT, vdotq_s32, coarse, no bias correction, bf16/f32/f16
 *  - SME: ARM fused BFMOPA, existing and unchanged
 */
#ifndef NUMKONG_MAXSIM_H
#define NUMKONG_MAXSIM_H

#include "numkong/types.h"
#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns packed buffer size in bytes for a maxsim vector set.
 *  @param[in] vector_count The number of vectors to pack.
 *  @param[in] depth The number of dimensions per vector.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the packed buffer.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_best(nk_size_t vector_count, nk_size_t depth,
                                                      nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_best(nk_size_t vector_count, nk_size_t depth,
                                                     nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_best(nk_size_t vector_count, nk_size_t depth,
                                                     nk_capability_t capabilities, nk_size_t *bytes);

/**
 *  @brief Reads a packed MaxSim buffer's shape from its header.
 *  @param[in] packed A buffer produced by the matching nk_maxsim_pack_bf16_best.
 *  @param[out] vectors Receives the vector count.
 *  @param[out] depth Receives the inner dimension.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_best(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                         nk_capability_t capabilities, void *stream);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_best(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_best(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Packs vectors into a backend-specific layout for maxsim computation.
 *  @param[in] vectors The input vectors in row-major order.
 *  @param[in] vector_count The number of vectors.
 *  @param[in] depth The number of dimensions per vector.
 *  @param[in] stride The row stride in bytes for the input vectors.
 *  @param[out] packed The output packed buffer from nk_maxsim_pack_size_bf16_best.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_best(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, void *packed, nk_capability_t capabilities,
                                                 void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_best(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                nk_size_t stride, void *packed, nk_capability_t capabilities,
                                                void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_best(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                nk_size_t stride, void *packed, nk_capability_t capabilities,
                                                void *stream);

/**
 *  @brief Computes angular distance late-interaction on pre-packed vectors. Returns Σᵢ minⱼ
 *      angular(qᵢ, dⱼ) where angular = 1 - dot / sqrt(||q||² × ||d||²).
 *
 *  @param[in] query_packed Packed query vectors (from nk_maxsim_pack_bf16_best).
 *  @param[in] document_packed Packed document vectors (from nk_maxsim_pack_bf16_best).
 *  @param[in] query_count Number of query vectors.
 *  @param[in] document_count Number of document vectors.
 *  @param[in] depth Number of dimensions per vector.
 *  @param[out] result Pointer to store the sum of per-query minimum angular distances.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_best(void const *query_packed, void const *document_packed,
                                                   nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                   nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_best(void const *query_packed, void const *document_packed,
                                                  nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                  nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_best(void const *query_packed, void const *document_packed,
                                                  nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                  nk_f32_t *result, nk_capability_t capabilities, void *stream);

/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_serial(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_serial(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_serial(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_serial(void const *query_packed, void const *document_packed,
                                                     nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                     nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_serial(void const *query_packed, void const *document_packed,
                                                    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                    nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_serial(void const *query_packed, void const *document_packed,
                                                    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                    nk_f32_t *result, void *stream);

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_icelake(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_icelake(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_icelake(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_icelake(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_icelake(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_icelake(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_icelake(void const *query_packed, void const *document_packed,
                                                     nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_icelake(void const *query_packed, void const *document_packed,
                                                     nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_genoa(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_genoa(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_genoa(void const *query_packed, void const *document_packed,
                                                    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                    nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_sapphireamx(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_sapphireamx(void const *packed, nk_size_t *vectors,
                                                                nk_size_t *depth, void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_sapphireamx(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_sapphireamx(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_sapphireamx(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_sapphireamx(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_sapphireamx(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_sapphireamx(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, void *packed, void *stream);

/**
 *  @copydoc nk_maxsim_packed_bf16_best
 *  @note Pipelines 4 document tiles through TMM4-7 with TDPBF16PS, gathering columns of the 16×16
 *      f32 accumulators into per-document dot products with AVX-512.
 */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_sapphireamx(void const *query_packed, void const *document_packed,
                                                          nk_size_t query_count, nk_size_t document_count,
                                                          nk_size_t depth, nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_sapphireamx(void const *query_packed, void const *document_packed,
                                                         nk_size_t query_count, nk_size_t document_count,
                                                         nk_size_t depth, nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_sapphireamx(void const *query_packed, void const *document_packed,
                                                         nk_size_t query_count, nk_size_t document_count,
                                                         nk_size_t depth, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_SAPPHIREAMX

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_haswell(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_haswell(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                   nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_haswell(void const *query_packed, void const *document_packed,
                                                      nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                      nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_haswell(void const *query_packed, void const *document_packed,
                                                     nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                     nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_haswell(void const *query_packed, void const *document_packed,
                                                     nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                     nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_ALDER
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                         void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                         void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_alder(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_alder(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_alder(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                 nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_alder(void const *query_packed, void const *document_packed,
                                                    nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                    nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_alder(void const *query_packed, void const *document_packed,
                                                   nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                   nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_alder(void const *query_packed, void const *document_packed,
                                                   nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                   nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_ALDER

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_v128relaxed(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_v128relaxed(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_v128relaxed(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_v128relaxed(void const *packed, nk_size_t *vectors,
                                                                nk_size_t *depth, void *stream);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_v128relaxed(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_v128relaxed(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
                                                        void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride_in_bytes, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride_in_bytes, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_v128relaxed(void const *query_packed, void const *document_packed,
                                                          nk_size_t query_count, nk_size_t document_count,
                                                          nk_size_t depth, nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_v128relaxed(void const *query_packed, void const *document_packed,
                                                         nk_size_t query_count, nk_size_t document_count,
                                                         nk_size_t depth, nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_v128relaxed(void const *query_packed, void const *document_packed,
                                                         nk_size_t query_count, nk_size_t document_count,
                                                         nk_size_t depth, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                             void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_neonsdot(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                     nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_neonsdot(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_neonsdot(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                    nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_neonsdot(void const *query_packed, void const *document_packed,
                                                       nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                       nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_neonsdot(void const *query_packed, void const *document_packed,
                                                      nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                      nk_f64_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_neonsdot(void const *query_packed, void const *document_packed,
                                                      nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                      nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_SME
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_sme(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                        void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_sme(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                       void *stream);
/** @copydoc nk_maxsim_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_sme(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes);
/** @copydoc nk_maxsim_packed_shape_f32_best */
NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_sme(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                       void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_bf16_sme(nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f16_sme(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                               nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_pack_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_pack_f32_sme(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                               nk_size_t stride, void *packed, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_bf16_sme(void const *query_packed, void const *document_packed,
                                                  nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                  nk_f32_t *result, void *stream);
/** @copydoc nk_maxsim_packed_bf16_best */
NUMKONG_API nk_status_t nk_maxsim_packed_f16_sme(void const *query_packed, void const *document_packed,
                                                 nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                 nk_f32_t *result, void *stream);

/**
 *  @copydoc nk_maxsim_packed_bf16_best
 *  @note Screens with i8 SMOPA, 4× the depth per instruction of f32 FMOPA, then refines the winning
 *      pairs in f64 into the angular distance 1 − dot / √(‖q‖² × ‖d‖²).
 */
NUMKONG_API nk_status_t nk_maxsim_packed_f32_sme(void const *query_packed, void const *document_packed,
                                                 nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                 nk_f64_t *result, void *stream);
#endif // NUMKONG_TARGET_SME

/** Returns the output dtype for MaxSim late-interaction. */
NUMKONG_INLINE nk_dtype_t nk_maxsim_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of MaxSim scores, per @c nk_accumulation_error_bound of their output. */
NUMKONG_INLINE nk_f64_t nk_maxsim_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_maxsim_output_dtype(dtype));
}

/**
 *  @brief Finds the MaxSim kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_maxsim_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/maxsim/serial.h"
#include "numkong/maxsim/haswell.h"
#include "numkong/maxsim/alder.h"
#include "numkong/maxsim/icelake.h"
#include "numkong/maxsim/genoa.h"
#include "numkong/maxsim/sapphireamx.h"
#include "numkong/maxsim/neonsdot.h"
#include "numkong/maxsim/sme.h"
#include "numkong/maxsim/v128.h"
#include "numkong/maxsim/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_best(nk_size_t width, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_best(void const *packed, nk_size_t *width, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(packed), nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_best(nk_f32_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                                void *b_packed, nk_capability_t capabilities, void *stream) {
    nk_unused_(b), nk_unused_(width), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_best(void const *q_packed, void const *d_packed, nk_size_t query_count,
                                                  nk_size_t document_count, nk_size_t depth, nk_f64_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_unused_(q_packed), nk_unused_(d_packed), nk_unused_(query_count), nk_unused_(document_count), nk_unused_(depth),
        nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_best(nk_size_t width, nk_size_t depth, nk_capability_t capabilities,
                                                      nk_size_t *bytes) {
    nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_best(void const *packed, nk_size_t *width, nk_size_t *depth,
                                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(packed), nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_best(nk_bf16_t const *b, nk_size_t width, nk_size_t depth,
                                                 nk_size_t b_stride, void *b_packed, nk_capability_t capabilities,
                                                 void *stream) {
    nk_unused_(b), nk_unused_(width), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_best(void const *q_packed, void const *d_packed, nk_size_t query_count,
                                                   nk_size_t document_count, nk_size_t depth, nk_f32_t *result,
                                                   nk_capability_t capabilities, void *stream) {
    nk_unused_(q_packed), nk_unused_(d_packed), nk_unused_(query_count), nk_unused_(document_count), nk_unused_(depth),
        nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_best(nk_size_t width, nk_size_t depth, nk_capability_t capabilities,
                                                     nk_size_t *bytes) {
    nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_best(void const *packed, nk_size_t *width, nk_size_t *depth,
                                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(packed), nk_unused_(width), nk_unused_(depth), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_best(nk_f16_t const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                                void *b_packed, nk_capability_t capabilities, void *stream) {
    nk_unused_(b), nk_unused_(width), nk_unused_(depth), nk_unused_(b_stride), nk_unused_(b_packed),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_best(void const *q_packed, void const *d_packed, nk_size_t query_count,
                                                  nk_size_t document_count, nk_size_t depth, nk_f32_t *result,
                                                  nk_capability_t capabilities, void *stream) {
    nk_unused_(q_packed), nk_unused_(d_packed), nk_unused_(query_count), nk_unused_(document_count), nk_unused_(depth),
        nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_maxsim_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_MAXSIM_H
