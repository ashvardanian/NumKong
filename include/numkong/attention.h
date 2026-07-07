/**
 *  @brief SIMD-accelerated ragged Transformer attention.
 *  @file include/numkong/attention.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  Contains the following kernel families, each with a size/pack/compute triple:
 *
 *  - `nk_attention_packed_size_<dtype>` - bytes needed to pack a ragged K/V batch
 *  - `nk_attention_pack_<dtype>` - one-time K/V packing into a backend-opaque layout
 *  - `nk_attention_<dtype>` - bidirectional scaled-dot-product attention over the batch
 *
 *  For dtypes:
 *
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *  - 8-bit `e4m3` floating point numbers → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - x86: Haswell, Skylake, Genoa, Sapphire Rapids AMX
 *  - Arm: SME
 *  - portable serial fallback
 *
 *  @section attention_usage Usage and Benefits
 *
 *  Transformer inference packs many variable-length segments into one flat token buffer;
 *  this family computes attention for the whole batch in one call, following UForm's
 *  `segment_offsets` / `segment_lengths` convention (cu_seqlens-style prefix sums).
 *  Self-attention, cross-attention, and single-query pooling are all the same kernel —
 *  only the per-segment query counts differ:
 *
 *  @code{.c}
 *  nk_u32_t offsets[] = {0, 100, 630, 663}, lengths[] = {100, 530, 33};   // 3 ragged segments
 *  nk_size_t bytes = nk_attention_packed_size_bf16(kv_heads, 128, lengths, 3);
 *  void *kv = aligned_alloc(64, bytes);
 *  nk_attention_pack_bf16(k, v, kv_heads, 128, offsets, lengths, 3, stride, stride, kv, 0, 0);
 *  nk_attention_packed_bf16(q, kv, out, heads, kv_heads, 128, offsets, stride, out_stride, scale, 0, 0);
 *  nk_u32_t one_each[] = {0, 1, 2, 3};  // attention pool: 1 learned query per segment
 *  nk_attention_packed_bf16(pool_q, kv, pooled, heads, kv_heads, 128, one_each, q_stride, o_stride, scale, 0, 0);
 *  @endcode
 *
 *  Q, K, V, O use the activations-natural `[tokens, heads × head_dim]` layout with byte
 *  strides, so a fused QKV projection output `[tokens, 3 × hidden]` is consumable in place.
 *  Both the packing and attention kernels accept a `(first_task, task_count)` window over
 *  the flat `segments × heads` grid (`(0, 0)` = everything); tasks touch disjoint outputs,
 *  so callers parallelize by distributing tasks across threads — one per physical core,
 *  longest segments first.  Outputs are F32: every consumer in a transformer block
 *  (normalization, residual epilogues) wants the accumulator precision anyway.
 *
 *  Unlike cuDNN's fused attention (head dims ≤ 256 and a multiple of 8 for 16-bit dtypes),
 *  any `head_dim ≥ 1` is supported: SIMD backends cover 1…256 with internal zero-padding,
 *  and larger head dimensions transparently fall back to the width-agnostic serial tier —
 *  the fallback rule is a pure function of the arguments, so packing and attention always
 *  agree on the buffer format.  Causal masking is deliberately absent: it belongs to a
 *  future decoder-shaped `nk_attention_causal_<dtype>` symbol, not to a flag that forks
 *  this hot loop.
 *
 *  @section attention_research Open Research Directions
 *
 *  Score-function replacements.  On AMX, tile registers support only load/store/zero and
 *  matrix-multiply — no elementwise ops — so any per-pair scoring function (softmax,
 *  sigmoid, ReLU²) forces the O(n²) score matrix through one memory→vector→memory round
 *  trip per panel.  Measured on one Sapphire Rapids core at q = kv = 1024, d = 128:
 *  softmax ≈ 0.82 TFLOPS, ReLU² ≈ 1.1 TFLOPS, sigmoid ≈ softmax (the division costs what
 *  the max/sum bookkeeping saves), and the tile ops alone ≈ 3.1 TFLOPS.  ReLU²-scored
 *  attention is the cheapest per-pair option, but has no known production deployments and
 *  requires training-time adoption with QK-norm, LayerScale, and 1/n scaling; validation
 *  loss does not predict its downstream failures, so retrieval-style probes are the gate.
 *  No zero-shot (quantization-style) softmax→ReLU² conversion exists; the nearest
 *  published path is a short annealing phase at the end of pretraining.
 *
 *  @see https://arxiv.org/abs/2309.08586 - ReLU/n scoring at softmax parity in ViTs
 *  @see https://arxiv.org/abs/2409.04431 - sigmoid attention theory; also benchmarks the
 *       ReLU² baselines and the QK-norm + LayerScale stabilizer stack
 *  @see https://arxiv.org/abs/2605.20798 - 1-3B replication of 20 modifications; documents
 *       the sigmoid retrieval collapse that validation loss never showed
 *  @see https://arxiv.org/abs/2410.18613 - polynomial substitutes for softmax, framed as
 *       Frobenius-norm regularization of the attention matrix
 *
 *  Linearized (kernelized) attention.  `O = φ(Q) · (φ(K)ᵀ V)` moves the nonlinearity from
 *  per-pair to per-token: the O(n·d) feature maps run on vector units while both
 *  contractions stay in the matrix unit, meeting at a d×d intermediate — the only
 *  attention class with no O(n²) tile↔vector crossover at all, and O(n·d²) complexity
 *  (≈128× less arithmetic at 16K tokens, d = 128).  Bottlenecks: quality at contrastive
 *  encoder scale is unproven; production encoder adoption is near zero; converting
 *  pretrained softmax checkpoints needs distillation (0.005-2% of pretraining tokens),
 *  never a gradient-free swap; and the d×d state must requantize to BF16 between the two
 *  matrix multiplications.
 *
 *  @see https://arxiv.org/abs/2402.05008 - EfficientViT-SAM, a shipped ReLU-kernel linear
 *       attention encoder at SAM-ViT-H quality
 *  @see https://arxiv.org/abs/2410.10254 - LoLCATs low-rank linearization of Llamas
 *  @see https://arxiv.org/abs/2505.03005 - RADLADS conversion at <0.005% of pretraining
 *  @see https://arxiv.org/abs/2402.04347 - Hedgehog: why zero-shot kernel swaps collapse
 *
 *  @section attention_references References
 *
 *  - FlashAttention-2 tiling and the online softmax: https://arxiv.org/abs/2307.08691
 *  - cuDNN attention shape constraints for comparison:
 *    https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef NK_ATTENTION_H
#define NK_ATTENTION_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns the packed KV-cache size in bytes for a ragged batch of segments.
 *  @param[in] num_kv_heads Number of K/V heads (≤ query heads for grouped-query attention).
 *  @param[in] head_dim Head dimension; any value ≥ 1.
 *  @param[in] segment_lengths Live token counts per segment, `[segment_count]`; zeros allowed.
 *  @param[in] segment_count Number of segments packed together.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_DYNAMIC nk_size_t nk_attention_packed_size_bf16(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                   nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_attention_packed_size_e4m3(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                   nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_DYNAMIC nk_size_t nk_attention_packed_size_i8(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                 nk_u32_t const *segment_lengths, nk_size_t segment_count);

/**
 *  @brief Packs a ragged batch of K and V segments into a backend-opaque layout.
 *  @param[in] k,v `[total_tokens, num_kv_heads × head_dim]` matrices with strided rows.
 *  @param[in] segment_offsets Start token of each segment, `[segment_count + 1]` prefix sums.
 *  @param[in] segment_lengths Live token counts, `[segment_count]`; zeros mark padding slots.
 *  @param[in] k_stride,v_stride Row (token) strides in bytes.
 *  @param[out] kv_packed 64-byte-aligned buffer of `nk_attention_packed_size_*` bytes.
 *  @param[in] first_task,task_count Window over the `segments × kv_heads` grid for parallel
 *      packing; `(0, 0)` packs everything.  Tasks write disjoint ranges; the tiny header and
 *      directory are written identically by every call.
 */
NK_DYNAMIC void nk_attention_pack_bf16(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                       nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                       nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
                                       nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_DYNAMIC void nk_attention_pack_e4m3(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                       nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                       nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
                                       nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_DYNAMIC void nk_attention_pack_i8(nk_i8_t const *k, nk_i8_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim,
                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                     nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                     nk_size_t first_task, nk_size_t task_count);

/**
 *  @brief Ragged bidirectional scaled-dot-product attention: `O[s] = softmax(Q[s]K[s]ᵀ·scale)V[s]`.
 *
 *  Covers self-attention (`query_offsets` equal to the pack-time `segment_offsets`),
 *  cross-attention, and pooling (`query_offsets = {0, 1, 2, …}` — one query per segment),
 *  plus GQA/MQA via `num_kv_heads < num_heads`.
 *
 *  @param[in] q `[total_query_tokens, num_heads × head_dim]` with `q_stride` bytes between rows.
 *  @param[in] kv_packed Buffer produced by the matching `nk_attention_pack_*` backend.
 *  @param[out] output `[total_query_tokens, num_heads × head_dim]` F32, `o_stride` bytes between rows.
 *  @param[in] query_offsets First query row of each segment, `[segment_count + 1]` prefix sums.
 *  @param[in] scale Score multiplier, typically `1 / sqrt(head_dim)`.
 *  @param[in] first_task,task_count Window over the `segments × heads` grid; `(0, 0)` runs all.
 *      Tasks write disjoint output regions, so callers parallelize across threads freely.
 */
NK_DYNAMIC void nk_attention_packed_bf16(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                         nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                         nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                         nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_DYNAMIC void nk_attention_packed_e4m3(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                         nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                         nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                         nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_DYNAMIC void nk_attention_packed_i8(nk_i8_t const *q, void const *kv_packed, nk_f32_t *output, nk_size_t num_heads,
                                       nk_size_t num_kv_heads, nk_size_t head_dim, nk_u32_t const *query_offsets,
                                       nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, nk_size_t first_task,
                                       nk_size_t task_count);

/** @brief Number of segments a packed KV buffer was built for. */
NK_PUBLIC nk_size_t nk_attention_packed_segments(void const *kv_packed);
/** @brief Head dimension a packed KV buffer was built for. */
NK_PUBLIC nk_size_t nk_attention_packed_head_dim(void const *kv_packed);
/** @brief Number of K/V heads a packed KV buffer was built for. */
NK_PUBLIC nk_size_t nk_attention_packed_heads(void const *kv_packed);

/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_i8_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_bf16_serial(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                             nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                             nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                             nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                             nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_e4m3_serial(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                             nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                             nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                             nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                             nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_i8_serial(nk_i8_t const *k, nk_i8_t const *v, nk_size_t num_kv_heads,
                                           nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                           nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
                                           nk_size_t v_stride, void *kv_packed, nk_size_t first_task,
                                           nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_bf16_serial(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                               nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                               nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                               nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_e4m3_serial(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                               nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                               nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                               nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_i8_serial(nk_i8_t const *q, void const *kv_packed, nk_f32_t *output,
                                             nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                             nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                             nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);

#if NK_TARGET_HASWELL
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_haswell(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_haswell(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_bf16_haswell(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                              nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                              nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                              nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_e4m3_haswell(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                              nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                              nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                              nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_bf16_haswell(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                                nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                                nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_e4m3_haswell(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                                nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                                nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_skylake(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_skylake(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_bf16_skylake(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                              nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                              nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                              nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_e4m3_skylake(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                              nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                              nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                              nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_bf16_skylake(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                                nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                                nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_e4m3_skylake(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                                nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                                nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_genoa(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_genoa(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_bf16_genoa(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                            nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                            nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                            nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_e4m3_genoa(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                            nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                            nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                            nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_bf16_genoa(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                              nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                              nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                              nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_e4m3_genoa(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                              nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                              nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                              nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_sapphireamx(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_bf16_sapphireamx(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                                  nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                                  nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                  nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                                  nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_bf16_sapphireamx(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                                    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                    nk_u32_t const *query_offsets, nk_size_t q_stride,
                                                    nk_size_t o_stride, nk_f32_t scale, nk_size_t first_task,
                                                    nk_size_t task_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_sapphireamx(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_e4m3_sapphireamx(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                                  nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                                  nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                  nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                                  nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_e4m3_sapphireamx(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                                    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                    nk_u32_t const *query_offsets, nk_size_t q_stride,
                                                    nk_size_t o_stride, nk_f32_t scale, nk_size_t first_task,
                                                    nk_size_t task_count);
/** @copydoc nk_attention_packed_size_bf16 */
NK_PUBLIC nk_size_t nk_attention_packed_size_i8_sapphireamx(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_bf16 */
NK_PUBLIC void nk_attention_pack_i8_sapphireamx(nk_i8_t const *k, nk_i8_t const *v, nk_size_t num_kv_heads,
                                                nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                                nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                                nk_size_t first_task, nk_size_t task_count);
/** @copydoc nk_attention_packed_bf16 */
NK_PUBLIC void nk_attention_packed_i8_sapphireamx(nk_i8_t const *q, void const *kv_packed, nk_f32_t *output,
                                                  nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                                  nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                                  nk_f32_t scale, nk_size_t first_task, nk_size_t task_count);
#endif // NK_TARGET_SAPPHIREAMX

/**
 *  @brief Returns the output dtype for attention: accumulator-precision F32 for all inputs.
 */
NK_INTERNAL nk_dtype_t nk_attention_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k; // quantized-weight softmax, F32 outputs
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/attention/serial.h"
#include "numkong/attention/haswell.h"
#include "numkong/attention/skylake.h"
#include "numkong/attention/genoa.h"
#include "numkong/attention/sapphireamx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                  nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_SAPPHIREAMX
    return nk_attention_packed_size_bf16_sapphireamx(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_GENOA
    return nk_attention_packed_size_bf16_genoa(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_SKYLAKE
    return nk_attention_packed_size_bf16_skylake(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_HASWELL
    return nk_attention_packed_size_bf16_haswell(num_kv_heads, head_dim, segment_lengths, segment_count);
#else
    return nk_attention_packed_size_bf16_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
#endif
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                  nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_SAPPHIREAMX
    return nk_attention_packed_size_e4m3_sapphireamx(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_GENOA
    return nk_attention_packed_size_e4m3_genoa(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_SKYLAKE
    return nk_attention_packed_size_e4m3_skylake(num_kv_heads, head_dim, segment_lengths, segment_count);
#elif NK_TARGET_HASWELL
    return nk_attention_packed_size_e4m3_haswell(num_kv_heads, head_dim, segment_lengths, segment_count);
#else
    return nk_attention_packed_size_e4m3_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
#endif
}

NK_PUBLIC void nk_attention_pack_bf16(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                      nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                      nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
                                      nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_pack_bf16_sapphireamx(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                       k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_GENOA
    nk_attention_pack_bf16_genoa(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                 k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_pack_bf16_skylake(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                   k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_pack_bf16_haswell(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                   k_stride, v_stride, kv_packed, first_task, task_count);
#else
    nk_attention_pack_bf16_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                  k_stride, v_stride, kv_packed, first_task, task_count);
#endif
}

NK_PUBLIC void nk_attention_pack_e4m3(nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads,
                                      nk_size_t head_dim, nk_u32_t const *segment_offsets,
                                      nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
                                      nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_pack_e4m3_sapphireamx(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                       k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_GENOA
    nk_attention_pack_e4m3_genoa(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                 k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_pack_e4m3_skylake(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                   k_stride, v_stride, kv_packed, first_task, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_pack_e4m3_haswell(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                   k_stride, v_stride, kv_packed, first_task, task_count);
#else
    nk_attention_pack_e4m3_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                  k_stride, v_stride, kv_packed, first_task, task_count);
#endif
}

NK_PUBLIC void nk_attention_packed_bf16(nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,
                                        nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                        nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                        nk_f32_t scale, nk_size_t first_task, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_packed_bf16_sapphireamx(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                         q_stride, o_stride, scale, first_task, task_count);
#elif NK_TARGET_GENOA
    nk_attention_packed_bf16_genoa(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                   o_stride, scale, first_task, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_packed_bf16_skylake(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                     o_stride, scale, first_task, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_packed_bf16_haswell(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                     o_stride, scale, first_task, task_count);
#else
    nk_attention_packed_bf16_serial(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                    o_stride, scale, first_task, task_count);
#endif
}

NK_PUBLIC void nk_attention_packed_e4m3(nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,
                                        nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,
                                        nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride,
                                        nk_f32_t scale, nk_size_t first_task, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_packed_e4m3_sapphireamx(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets,
                                         q_stride, o_stride, scale, first_task, task_count);
#elif NK_TARGET_GENOA
    nk_attention_packed_e4m3_genoa(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                   o_stride, scale, first_task, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_packed_e4m3_skylake(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                     o_stride, scale, first_task, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_packed_e4m3_haswell(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                     o_stride, scale, first_task, task_count);
#else
    nk_attention_packed_e4m3_serial(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                    o_stride, scale, first_task, task_count);
#endif
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_SAPPHIREAMX
    return nk_attention_packed_size_i8_sapphireamx(num_kv_heads, head_dim, segment_lengths, segment_count);
#else
    return nk_attention_packed_size_i8_serial(num_kv_heads, head_dim, segment_lengths, segment_count);
#endif
}

NK_PUBLIC void nk_attention_pack_i8(nk_i8_t const *k, nk_i8_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim,
                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                    nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed,
                                    nk_size_t first_task, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_pack_i8_sapphireamx(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count,
                                     k_stride, v_stride, kv_packed, first_task, task_count);
#else
    nk_attention_pack_i8_serial(k, v, num_kv_heads, head_dim, segment_offsets, segment_lengths, segment_count, k_stride,
                                v_stride, kv_packed, first_task, task_count);
#endif
}

NK_PUBLIC void nk_attention_packed_i8(nk_i8_t const *q, void const *kv_packed, nk_f32_t *output, nk_size_t num_heads,
                                      nk_size_t num_kv_heads, nk_size_t head_dim, nk_u32_t const *query_offsets,
                                      nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, nk_size_t first_task,
                                      nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_packed_i8_sapphireamx(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                       o_stride, scale, first_task, task_count);
#else
    nk_attention_packed_i8_serial(q, kv_packed, output, num_heads, num_kv_heads, head_dim, query_offsets, q_stride,
                                  o_stride, scale, first_task, task_count);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_ATTENTION_H
