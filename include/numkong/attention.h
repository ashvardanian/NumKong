/**
 *  @brief SIMD-accelerated ragged Transformer attention.
 *  @file include/numkong/attention.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  Contains the following kernel families, each with a size/pack/compute triple:
 *
 *  - `nk_attention_pack_size_<dtype>` - bytes needed to pack a ragged K/V batch
 *  - `nk_attention_pack_<dtype>` - one-time K/V packing into a backend-opaque layout
 *  - `nk_attention_bidirectional_packed_<dtype>` - bidirectional scaled-dot-product attention over the batch
 *  - `nk_attention_causal_packed_<dtype>` - causal, optionally sliding-window, attention over the same pack
 *
 *  For dtypes:
 *
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *  - 8-bit `e4m3` floating point numbers → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - x86: Haswell, Skylake, Genoa, Sapphire Rapids AMX
 *  - Arm: SME, and per-feature NEON tiers (BFDOT for BF16, FHM for E4M3, SDOT for I8)
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
 *  nk_size_t bytes = nk_attention_pack_size_bf16(kv_heads, 128, lengths, 3);
 *  void *kv = aligned_alloc(64, bytes);
 *  nk_attention_pack_bf16(keys, values, kv_heads, 128, offsets, lengths, 3, stride, stride, kv, 0, 3 * kv_heads);
 *  nk_attention_bidirectional_packed_bf16(queries, kv, out, heads, kv_heads, 128, offsets, stride, out_stride,
 *                                         scale, 0, NK_SIZE_MAX);
 *  nk_u32_t one_each[] = {0, 1, 2, 3};  // attention pool: 1 learned query per segment
 *  nk_attention_bidirectional_packed_bf16(pool_q, kv, pooled, heads, kv_heads, 128, one_each, stride, out_stride,
 *                                         scale, 0, NK_SIZE_MAX);
 *  @endcode
 *
 *  Q, K, V, O use the activations-natural `[tokens, heads × depth]` layout with byte
 *  strides, so a fused QKV projection output `[tokens, 3 × hidden]` is consumable in place.
 *  Packing takes a half-open `(task_begin, task_end)` window over the flat `segments × kv_heads` grid,
 *  and attention a `(task_start, task_count)` window over `segments × heads`; tasks touch disjoint outputs,
 *  so callers parallelize by distributing tasks across threads — one per physical core,
 *  longest segments first. Outputs are F32: every consumer in a transformer block
 *  (normalization, residual epilogues) wants the accumulator precision anyway.
 *
 *  Unlike cuDNN's fused attention (head dims ≤ 256 and a multiple of 8 for 16-bit dtypes),
 *  any `depth ≥ 1` is supported: SIMD backends cover 1…256 with internal zero-padding,
 *  and larger head dimensions transparently fall back to the width-agnostic serial tier —
 *  the fallback rule is a pure function of the arguments, so packing and attention always
 *  agree on the buffer format. Causal masking lives in `nk_attention_causal_packed_<dtype>`,
 *  a separate symbol over the same pack, rather than a flag on the bidirectional kernel.
 *
 *  @section attention_research Open Research Directions
 *
 *  Score-function replacements. On AMX, tile registers support only load/store/zero and
 *  matrix-multiply — no elementwise ops — so any per-pair scoring function (softmax,
 *  sigmoid, ReLU²) forces the O(n²) score matrix through one memory→vector→memory round
 *  trip per panel. Measured on one Sapphire Rapids core at q = kv = 1024, d = 128:
 *  softmax ≈ 0.82 TFLOPS, ReLU² ≈ 1.1 TFLOPS, sigmoid ≈ softmax (the division costs what
 *  the max/sum bookkeeping saves), and the tile ops alone ≈ 3.1 TFLOPS. ReLU²-scored
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
 *  Linearized (kernelized) attention. `O = φ(Q) · (φ(K)ᵀ V)` moves the nonlinearity from
 *  per-pair to per-token: the O(n·d) feature maps run on vector units while both
 *  contractions stay in the matrix unit, meeting at a d×d intermediate — the only
 *  attention class with no O(n²) tile↔vector crossover at all, and O(n·d²) complexity
 *  (≈128× less arithmetic at 16K tokens, d = 128). Bottlenecks: quality at contrastive
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
 *  @section attention_causal Causal and Sliding-Window Attention
 *
 *  `nk_attention_causal_packed_*` covers decoder inference with two scalars: query row `r` sits at
 *  position `p = r + diagonal_offset` and sees the `window` keys ending at `p`, inclusive. `offset = 0`
 *  gives causal prefill, `offset = position_count − row_count` gives decode and chunked prefill against a
 *  longer cache, a finite `window` gives sliding-window attention, and the ragged segment directory
 *  already provides block-diagonal document masking for packed batches. Rows whose range is empty,
 *  such as `p < 0` or `window = 0`, produce zeros, as do segments with no keys in both modes.
 *
 *  Masking clips each row's key range instead of writing −∞ scores, because the clamped `exp2`,
 *  SME rounding, and I8 score differences all misbehave on sentinels. KV panels outside every row of a
 *  query block are skipped outright, so causality is ≈2× fewer FLOPs at equal context, and a row that
 *  sees no key of a panel keeps its running maximum. Deliberately out of scope: ALiBi, logit
 *  soft-capping, and paged KV caches.
 *
 *  @see https://arxiv.org/abs/2307.08691 - FlashAttention-2 causal tiling and work skipping
 *  @see https://arxiv.org/abs/2310.06825 - Mistral 7B: sliding-window attention in production
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
 *  @param[in] key_value_head_count Number of K/V heads (≤ query heads for grouped-query attention).
 *  @param[in] depth Head dimension; any value ≥ 1.
 *  @param[in] segment_lengths Live token counts per segment, `[segment_count]`; zeros allowed.
 *  @param[in] segment_count Number of segments packed together.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NK_API_RUNTIME nk_size_t nk_attention_pack_size_bf16(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_RUNTIME nk_size_t nk_attention_pack_size_e4m3(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_RUNTIME nk_size_t nk_attention_pack_size_i8(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_lengths, nk_size_t segment_count);

/**
 *  @brief Reads a packed KV cache's shape from its header.
 *  @param[in] key_value_packed A buffer produced by the matching nk_attention_pack_bf16.
 *  @param[out] heads Receives the K/V head count.
 *  @param[out] depth Receives the head dimension.
 *  @param[out] segments Receives the segment count.
 */
NK_API_RUNTIME void nk_attention_packed_shape_bf16(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                   nk_size_t *segments);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_RUNTIME void nk_attention_packed_shape_e4m3(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                   nk_size_t *segments);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_RUNTIME void nk_attention_packed_shape_i8(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                 nk_size_t *segments);

/**
 *  @brief Packs a ragged batch of K and V segments into a backend-opaque layout.
 *  @param[in] keys,values `[total_tokens, key_value_head_count × depth]` matrices with strided rows.
 *  @param[in] segment_offsets Start token of each segment, `[segment_count + 1]` prefix sums.
 *  @param[in] segment_lengths Live token counts, `[segment_count]`; zeros mark padding slots.
 *  @param[in] key_stride_bytes,value_stride_bytes Row (token) strides in bytes.
 *  @param[out] key_value_packed 64-byte-aligned buffer of `nk_attention_pack_size_*` bytes.
 *  @param[in] task_begin,task_end Half-open window over the `segments × kv_heads` grid for parallel
 *      packing, with `task_end` clipped to the grid. Tasks write disjoint ranges; the header and
 *      directory are written by the window starting at task 0.
 */
NK_API_RUNTIME void nk_attention_pack_bf16(nk_bf16_t const *keys, nk_bf16_t const *values,
                                           nk_size_t key_value_head_count, nk_size_t depth,
                                           nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                           nk_size_t segment_count, nk_size_t key_stride_bytes,
                                           nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
                                           nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_RUNTIME void nk_attention_pack_e4m3(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                           nk_size_t key_value_head_count, nk_size_t depth,
                                           nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                           nk_size_t segment_count, nk_size_t key_stride_bytes,
                                           nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
                                           nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_RUNTIME void nk_attention_pack_i8(nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count,
                                         nk_size_t depth, nk_u32_t const *segment_offsets,
                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                         nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                         void *key_value_packed, nk_size_t task_begin, nk_size_t task_end);

/**
 *  @brief Ragged bidirectional scaled-dot-product attention: `O[s] = softmax(Q[s]K[s]ᵀ·scale)V[s]`.
 *
 *  Covers self-attention (`query_offsets` equal to the pack-time `segment_offsets`),
 *  cross-attention, and pooling (`query_offsets = {0, 1, 2, …}` — one query per segment),
 *  plus GQA/MQA via `key_value_head_count < head_count`.
 *
 *  @param[in] queries `[total_query_tokens, head_count × depth]` with `query_stride_bytes` bytes between rows.
 *  @param[in] key_value_packed Buffer produced by the matching `nk_attention_pack_*` backend.
 *  @param[out] output `[total_query_tokens, head_count × depth]` F32, `output_stride_bytes` bytes between rows.
 *  @param[in] query_offsets First query row of each segment, `[segment_count + 1]` prefix sums.
 *  @param[in] scale Score multiplier, typically `1 / sqrt(depth)`.
 *  @param[in] task_start,task_count Window over the `segments × heads` grid, with `task_count` clipped to
 *      `segments · heads − task_start`. Tasks write disjoint output regions, so callers parallelize freely.
 */
NK_API_RUNTIME void nk_attention_bidirectional_packed_bf16(nk_bf16_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_size_t task_start, nk_size_t task_count);
/**
 *  @brief Ragged causal scaled-dot-product attention with an optional sliding window.
 *
 *  Query row `r` of a segment sits at position `p = r + diagonal_offset` and attends to the keys
 *  `[max(0, p − window + 1), min(p, length − 1)]`; rows with an empty range produce zeros.
 *  Packing is shared with `nk_attention_bidirectional_packed_bf16`, as are all other parameters.
 *
 *  @param[in] diagonal_offset Position of query row 0: `0` for prefill, `length − query_count` against a cache.
 *  @param[in] window Visible keys including the query itself; `NK_SIZE_MAX` is unbounded, `0` masks every key.
 */
NK_API_RUNTIME void nk_attention_causal_packed_bf16(nk_bf16_t const *queries, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t head_count,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                    nk_size_t output_stride_bytes, nk_f32_t scale,
                                                    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_RUNTIME void nk_attention_bidirectional_packed_e4m3(nk_e4m3_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_RUNTIME void nk_attention_causal_packed_e4m3(nk_e4m3_t const *queries, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t head_count,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                    nk_size_t output_stride_bytes, nk_f32_t scale,
                                                    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_RUNTIME void nk_attention_bidirectional_packed_i8(nk_i8_t const *queries, void const *key_value_packed,
                                                         nk_f32_t *output, nk_size_t head_count,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                         nk_size_t output_stride_bytes, nk_f32_t scale,
                                                         nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_RUNTIME void nk_attention_causal_packed_i8(nk_i8_t const *queries, void const *key_value_packed,
                                                  nk_f32_t *output, nk_size_t head_count,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                  nk_size_t output_stride_bytes, nk_f32_t scale,
                                                  nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                  nk_size_t task_count);

/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_serial(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_serial(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_serial(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_serial(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_serial(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_serial(nk_i8_t const *keys, nk_i8_t const *values,
                                                 nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                 nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                 nk_size_t value_stride_bytes, void *key_value_packed,
                                                 nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_serial(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_serial(nk_bf16_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_serial(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_serial(nk_e4m3_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_serial(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_serial(nk_i8_t const *queries, void const *key_value_packed,
                                                          nk_f32_t *output, nk_size_t head_count,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                          nk_size_t output_stride_bytes, nk_f32_t scale,
                                                          nk_i64_t diagonal_offset, nk_size_t window,
                                                          nk_size_t task_start, nk_size_t task_count);

#if NK_TARGET_HASWELL
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_haswell(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_haswell(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_haswell(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_haswell(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_haswell(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_haswell(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_haswell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_haswell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_haswell(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_haswell(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_haswell(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_haswell(nk_i8_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_skylake(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_skylake(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_skylake(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_skylake(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_skylake(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_skylake(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_skylake(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_skylake(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_ICELAKE
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_icelake(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_icelake(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_icelake(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_icelake(nk_i8_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_GENOA
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_genoa(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_genoa(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_genoa(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_genoa(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_genoa(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_genoa(nk_bf16_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_genoa(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_genoa(nk_e4m3_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count);
#endif // NK_TARGET_GENOA

#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                                nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_sapphireamx(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                        nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_offsets,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                        void *key_value_packed, nk_size_t task_begin,
                                                        nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_sapphireamx(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_sapphireamx(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                                nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_sapphireamx(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                        nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_offsets,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                        void *key_value_packed, nk_size_t task_begin,
                                                        nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_sapphireamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_sapphireamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_sapphireamx(nk_i8_t const *keys, nk_i8_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_sapphireamx(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_sapphireamx(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_SAPPHIREAMX

#if NK_TARGET_DIAMONDAMX
/* Diamond Rapids AMX provides only the E4M3 attention variant: its native FP8 tiles (`_tile_dphf8ps`)
 * are its differentiator, while its I8/BF16 paths would merely clone the Sapphire AMX backend. */
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_u32_t const *segment_lengths,
                                                                 nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_diamondamx(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_diamondamx(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_diamondamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_diamondamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_DIAMONDAMX

#if NK_TARGET_SME
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_sme(void const *key_value_packed, nk_size_t *heads,
                                                        nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_sme(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                nk_size_t value_stride_bytes, void *key_value_packed,
                                                nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_sme(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_sme(nk_bf16_t const *queries, void const *key_value_packed,
                                                         nk_f32_t *output, nk_size_t head_count,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                         nk_size_t output_stride_bytes, nk_f32_t scale,
                                                         nk_i64_t diagonal_offset, nk_size_t window,
                                                         nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_sme(void const *key_value_packed, nk_size_t *heads,
                                                        nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_sme(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                nk_size_t value_stride_bytes, void *key_value_packed,
                                                nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_sme(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_sme(nk_e4m3_t const *queries, void const *key_value_packed,
                                                         nk_f32_t *output, nk_size_t head_count,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                         nk_size_t output_stride_bytes, nk_f32_t scale,
                                                         nk_i64_t diagonal_offset, nk_size_t window,
                                                         nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_sme(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                      nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_sme(nk_i8_t const *keys, nk_i8_t const *values,
                                              nk_size_t key_value_head_count, nk_size_t depth,
                                              nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                              nk_size_t segment_count, nk_size_t key_stride_bytes,
                                              nk_size_t value_stride_bytes, void *key_value_packed,
                                              nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_sme(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_sme(nk_i8_t const *queries, void const *key_value_packed,
                                                       nk_f32_t *output, nk_size_t head_count,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                       nk_size_t output_stride_bytes, nk_f32_t scale,
                                                       nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                       nk_size_t task_count);
#endif // NK_TARGET_SME

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_neonbfdot(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_neonbfdot(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_neonbfdot(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_neonbfdot(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_NEONFHM
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_neonfhm(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_neonfhm(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_neonfhm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_neonfhm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_NEONFHM

#if NK_TARGET_NEONSDOT
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_neonsdot(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_neonsdot(nk_i8_t const *keys, nk_i8_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_neonsdot(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_neonsdot(nk_i8_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count);
#endif // NK_TARGET_NEONSDOT

#if NK_TARGET_RVV
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_rvv(void const *key_value_packed, nk_size_t *heads,
                                                        nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_rvv(void const *key_value_packed, nk_size_t *heads,
                                                        nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_rvv(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                      nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_rvv(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                nk_size_t value_stride_bytes, void *key_value_packed,
                                                nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_rvv(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                nk_size_t key_value_head_count, nk_size_t depth,
                                                nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                nk_size_t value_stride_bytes, void *key_value_packed,
                                                nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_rvv(nk_i8_t const *keys, nk_i8_t const *values,
                                              nk_size_t key_value_head_count, nk_size_t depth,
                                              nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                              nk_size_t segment_count, nk_size_t key_stride_bytes,
                                              nk_size_t value_stride_bytes, void *key_value_packed,
                                              nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_rvv(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_rvv(nk_bf16_t const *queries, void const *key_value_packed,
                                                         nk_f32_t *output, nk_size_t head_count,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                         nk_size_t output_stride_bytes, nk_f32_t scale,
                                                         nk_i64_t diagonal_offset, nk_size_t window,
                                                         nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_rvv(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_rvv(nk_e4m3_t const *queries, void const *key_value_packed,
                                                         nk_f32_t *output, nk_size_t head_count,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                         nk_size_t output_stride_bytes, nk_f32_t scale,
                                                         nk_i64_t diagonal_offset, nk_size_t window,
                                                         nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_rvv(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_rvv(nk_i8_t const *queries, void const *key_value_packed,
                                                       nk_f32_t *output, nk_size_t head_count,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                       nk_size_t output_stride_bytes, nk_f32_t scale,
                                                       nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                       nk_size_t task_count);
#endif // NK_TARGET_RVV

#if NK_TARGET_V128
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_bf16 */
NK_API_COMPTIME void nk_attention_packed_shape_bf16_v128(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_e4m3 */
NK_API_COMPTIME void nk_attention_packed_shape_e4m3_v128(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments);
/** @copydoc nk_attention_pack_size_bf16 */
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count);
/** @copydoc nk_attention_packed_shape_i8 */
NK_API_COMPTIME void nk_attention_packed_shape_i8_v128(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                       nk_size_t *segments);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_bf16_v128(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                 nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                 nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                 nk_size_t value_stride_bytes, void *key_value_packed,
                                                 nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_e4m3_v128(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                 nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                 nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                 nk_size_t value_stride_bytes, void *key_value_packed,
                                                 nk_size_t task_begin, nk_size_t task_end);
/** @copydoc nk_attention_pack_bf16 */
NK_API_COMPTIME void nk_attention_pack_i8_v128(nk_i8_t const *keys, nk_i8_t const *values,
                                               nk_size_t key_value_head_count, nk_size_t depth,
                                               nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                               nk_size_t segment_count, nk_size_t key_stride_bytes,
                                               nk_size_t value_stride_bytes, void *key_value_packed,
                                               nk_size_t task_begin, nk_size_t task_end);
#endif // NK_TARGET_V128

#if NK_TARGET_V128RELAXED
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_v128relaxed(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_bf16_v128relaxed(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_v128relaxed(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_e4m3_v128relaxed(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
/** @copydoc nk_attention_bidirectional_packed_bf16 */
NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_v128relaxed(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count);
/** @copydoc nk_attention_causal_packed_bf16 */
NK_API_COMPTIME void nk_attention_causal_packed_i8_v128relaxed(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count);
#endif // NK_TARGET_V128RELAXED

/**
 *  @brief Returns the output dtype for attention: accumulator-precision F32 for all inputs.
 */
NK_HELPER_INLINE nk_dtype_t nk_attention_output_dtype(nk_dtype_t dtype) {
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
#include "numkong/attention/icelake.h"
#include "numkong/attention/genoa.h"
#include "numkong/attention/sapphireamx.h"
#include "numkong/attention/diamondamx.h"
#include "numkong/attention/sme.h"
#include "numkong/attention/neonbfdot.h"
#include "numkong/attention/neonfhm.h"
#include "numkong/attention/neonsdot.h"
#include "numkong/attention/rvv.h"
#include "numkong/attention/v128.h"
#include "numkong/attention/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_RUNTIME_DISPATCH

NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_SAPPHIREAMX
    return nk_attention_pack_size_bf16_sapphireamx(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_GENOA
    return nk_attention_pack_size_bf16_genoa(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SKYLAKE
    return nk_attention_pack_size_bf16_skylake(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_HASWELL
    return nk_attention_pack_size_bf16_haswell(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SME
    return nk_attention_pack_size_bf16_sme(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_NEONBFDOT
    return nk_attention_pack_size_bf16_neonbfdot(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_RVV
    return nk_attention_pack_size_bf16_rvv(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_V128
    return nk_attention_pack_size_bf16_v128(key_value_head_count, depth, segment_lengths, segment_count);
#else
    return nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
#endif
}

NK_API_COMPTIME void nk_attention_packed_shape_bf16(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                    nk_size_t *segments) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_packed_shape_bf16_sapphireamx(key_value_packed, heads, depth, segments);
#elif NK_TARGET_GENOA
    nk_attention_packed_shape_bf16_genoa(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SKYLAKE
    nk_attention_packed_shape_bf16_skylake(key_value_packed, heads, depth, segments);
#elif NK_TARGET_HASWELL
    nk_attention_packed_shape_bf16_haswell(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SME
    nk_attention_packed_shape_bf16_sme(key_value_packed, heads, depth, segments);
#elif NK_TARGET_NEONBFDOT
    nk_attention_packed_shape_bf16_neonbfdot(key_value_packed, heads, depth, segments);
#elif NK_TARGET_RVV
    nk_attention_packed_shape_bf16_rvv(key_value_packed, heads, depth, segments);
#elif NK_TARGET_V128
    nk_attention_packed_shape_bf16_v128(key_value_packed, heads, depth, segments);
#else
    nk_attention_packed_shape_bf16_serial(key_value_packed, heads, depth, segments);
#endif
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_DIAMONDAMX
    return nk_attention_pack_size_e4m3_diamondamx(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SAPPHIREAMX
    return nk_attention_pack_size_e4m3_sapphireamx(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_GENOA
    return nk_attention_pack_size_e4m3_genoa(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SKYLAKE
    return nk_attention_pack_size_e4m3_skylake(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_HASWELL
    return nk_attention_pack_size_e4m3_haswell(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SME
    return nk_attention_pack_size_e4m3_sme(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_NEONFHM
    return nk_attention_pack_size_e4m3_neonfhm(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_RVV
    return nk_attention_pack_size_e4m3_rvv(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_V128
    return nk_attention_pack_size_e4m3_v128(key_value_head_count, depth, segment_lengths, segment_count);
#else
    return nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
#endif
}

NK_API_COMPTIME void nk_attention_packed_shape_e4m3(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                    nk_size_t *segments) {
#if NK_TARGET_DIAMONDAMX
    nk_attention_packed_shape_e4m3_diamondamx(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SAPPHIREAMX
    nk_attention_packed_shape_e4m3_sapphireamx(key_value_packed, heads, depth, segments);
#elif NK_TARGET_GENOA
    nk_attention_packed_shape_e4m3_genoa(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SKYLAKE
    nk_attention_packed_shape_e4m3_skylake(key_value_packed, heads, depth, segments);
#elif NK_TARGET_HASWELL
    nk_attention_packed_shape_e4m3_haswell(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SME
    nk_attention_packed_shape_e4m3_sme(key_value_packed, heads, depth, segments);
#elif NK_TARGET_NEONFHM
    nk_attention_packed_shape_e4m3_neonfhm(key_value_packed, heads, depth, segments);
#elif NK_TARGET_RVV
    nk_attention_packed_shape_e4m3_rvv(key_value_packed, heads, depth, segments);
#elif NK_TARGET_V128
    nk_attention_packed_shape_e4m3_v128(key_value_packed, heads, depth, segments);
#else
    nk_attention_packed_shape_e4m3_serial(key_value_packed, heads, depth, segments);
#endif
}

NK_API_COMPTIME void nk_attention_pack_bf16(nk_bf16_t const *keys, nk_bf16_t const *values,
                                            nk_size_t key_value_head_count, nk_size_t depth,
                                            nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                            nk_size_t segment_count, nk_size_t key_stride_bytes,
                                            nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
                                            nk_size_t task_end) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_pack_bf16_sapphireamx(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                       segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                                       task_begin, task_end);
#elif NK_TARGET_GENOA
    nk_attention_pack_bf16_genoa(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                 segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                 task_end);
#elif NK_TARGET_SKYLAKE
    nk_attention_pack_bf16_skylake(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                   task_end);
#elif NK_TARGET_HASWELL
    nk_attention_pack_bf16_haswell(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                   task_end);
#elif NK_TARGET_SME
    nk_attention_pack_bf16_sme(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                               task_end);
#elif NK_TARGET_NEONBFDOT
    nk_attention_pack_bf16_neonbfdot(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                     segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                     task_end);
#elif NK_TARGET_RVV
    nk_attention_pack_bf16_rvv(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                               task_end);
#elif NK_TARGET_V128
    nk_attention_pack_bf16_v128(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                task_end);
#else
    nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                  segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                  task_end);
#endif
}

NK_API_COMPTIME void nk_attention_pack_e4m3(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                            nk_size_t key_value_head_count, nk_size_t depth,
                                            nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                            nk_size_t segment_count, nk_size_t key_stride_bytes,
                                            nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
                                            nk_size_t task_end) {
#if NK_TARGET_DIAMONDAMX
    nk_attention_pack_e4m3_diamondamx(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                      task_end);
#elif NK_TARGET_SAPPHIREAMX
    nk_attention_pack_e4m3_sapphireamx(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                       segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                                       task_begin, task_end);
#elif NK_TARGET_GENOA
    nk_attention_pack_e4m3_genoa(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                 segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                 task_end);
#elif NK_TARGET_SKYLAKE
    nk_attention_pack_e4m3_skylake(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                   task_end);
#elif NK_TARGET_HASWELL
    nk_attention_pack_e4m3_haswell(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                   task_end);
#elif NK_TARGET_SME
    nk_attention_pack_e4m3_sme(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                               task_end);
#elif NK_TARGET_NEONFHM
    nk_attention_pack_e4m3_neonfhm(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                   task_end);
#elif NK_TARGET_RVV
    nk_attention_pack_e4m3_rvv(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                               task_end);
#elif NK_TARGET_V128
    nk_attention_pack_e4m3_v128(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                task_end);
#else
    nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                  segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                  task_end);
#endif
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16(nk_bf16_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_size_t task_start, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_bidirectional_packed_bf16_sapphireamx(queries, key_value_packed, output, head_count,
                                                       key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                       output_stride_bytes, scale, task_start, task_count);
#elif NK_TARGET_GENOA
    nk_attention_bidirectional_packed_bf16_genoa(queries, key_value_packed, output, head_count, key_value_head_count,
                                                 depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                 task_start, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_bidirectional_packed_bf16_skylake(queries, key_value_packed, output, head_count, key_value_head_count,
                                                   depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                   task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_bidirectional_packed_bf16_haswell(queries, key_value_packed, output, head_count, key_value_head_count,
                                                   depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                   task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_bidirectional_packed_bf16_sme(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               task_start, task_count);
#elif NK_TARGET_NEONBFDOT
    nk_attention_bidirectional_packed_bf16_neonbfdot(queries, key_value_packed, output, head_count,
                                                     key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                     output_stride_bytes, scale, task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_bidirectional_packed_bf16_rvv(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               task_start, task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_bidirectional_packed_bf16_v128relaxed(queries, key_value_packed, output, head_count,
                                                       key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                       output_stride_bytes, scale, task_start, task_count);
#else
    nk_attention_bidirectional_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count,
                                                  depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                  task_start, task_count);
#endif
}

NK_API_COMPTIME void nk_attention_causal_packed_bf16(nk_bf16_t const *queries, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                     nk_size_t output_stride_bytes, nk_f32_t scale,
                                                     nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                     nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_causal_packed_bf16_sapphireamx(queries, key_value_packed, output, head_count, key_value_head_count,
                                                depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_GENOA
    nk_attention_causal_packed_bf16_genoa(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                          query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                          diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_causal_packed_bf16_skylake(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                            query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                            diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_causal_packed_bf16_haswell(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                            query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                            diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_causal_packed_bf16_sme(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                        window, task_start, task_count);
#elif NK_TARGET_NEONBFDOT
    nk_attention_causal_packed_bf16_neonbfdot(queries, key_value_packed, output, head_count, key_value_head_count,
                                              depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                              diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_causal_packed_bf16_rvv(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                        window, task_start, task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_causal_packed_bf16_v128relaxed(queries, key_value_packed, output, head_count, key_value_head_count,
                                                depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                diagonal_offset, window, task_start, task_count);
#else
    nk_attention_causal_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                           query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                           diagonal_offset, window, task_start, task_count);
#endif
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3(nk_e4m3_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_size_t task_start, nk_size_t task_count) {
#if NK_TARGET_DIAMONDAMX
    nk_attention_bidirectional_packed_e4m3_diamondamx(queries, key_value_packed, output, head_count,
                                                      key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                      output_stride_bytes, scale, task_start, task_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_attention_bidirectional_packed_e4m3_sapphireamx(queries, key_value_packed, output, head_count,
                                                       key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                       output_stride_bytes, scale, task_start, task_count);
#elif NK_TARGET_GENOA
    nk_attention_bidirectional_packed_e4m3_genoa(queries, key_value_packed, output, head_count, key_value_head_count,
                                                 depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                 task_start, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_bidirectional_packed_e4m3_skylake(queries, key_value_packed, output, head_count, key_value_head_count,
                                                   depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                   task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_bidirectional_packed_e4m3_haswell(queries, key_value_packed, output, head_count, key_value_head_count,
                                                   depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                   task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_bidirectional_packed_e4m3_sme(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               task_start, task_count);
#elif NK_TARGET_NEONFHM
    nk_attention_bidirectional_packed_e4m3_neonfhm(queries, key_value_packed, output, head_count, key_value_head_count,
                                                   depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                   task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_bidirectional_packed_e4m3_rvv(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               task_start, task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_bidirectional_packed_e4m3_v128relaxed(queries, key_value_packed, output, head_count,
                                                       key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                       output_stride_bytes, scale, task_start, task_count);
#else
    nk_attention_bidirectional_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count,
                                                  depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                  task_start, task_count);
#endif
}

NK_API_COMPTIME void nk_attention_causal_packed_e4m3(nk_e4m3_t const *queries, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t head_count,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                     nk_size_t output_stride_bytes, nk_f32_t scale,
                                                     nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                     nk_size_t task_count) {
#if NK_TARGET_DIAMONDAMX
    nk_attention_causal_packed_e4m3_diamondamx(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_attention_causal_packed_e4m3_sapphireamx(queries, key_value_packed, output, head_count, key_value_head_count,
                                                depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_GENOA
    nk_attention_causal_packed_e4m3_genoa(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                          query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                          diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SKYLAKE
    nk_attention_causal_packed_e4m3_skylake(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                            query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                            diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_causal_packed_e4m3_haswell(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                            query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                            diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_causal_packed_e4m3_sme(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                        window, task_start, task_count);
#elif NK_TARGET_NEONFHM
    nk_attention_causal_packed_e4m3_neonfhm(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                            query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                            diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_causal_packed_e4m3_rvv(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                        window, task_start, task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_causal_packed_e4m3_v128relaxed(queries, key_value_packed, output, head_count, key_value_head_count,
                                                depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                diagonal_offset, window, task_start, task_count);
#else
    nk_attention_causal_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                           query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                           diagonal_offset, window, task_start, task_count);
#endif
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_lengths, nk_size_t segment_count) {
#if NK_TARGET_SAPPHIREAMX
    return nk_attention_pack_size_i8_sapphireamx(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_ICELAKE
    return nk_attention_pack_size_i8_icelake(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_HASWELL
    return nk_attention_pack_size_i8_haswell(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_SME
    return nk_attention_pack_size_i8_sme(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_NEONSDOT
    return nk_attention_pack_size_i8_neonsdot(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_RVV
    return nk_attention_pack_size_i8_rvv(key_value_head_count, depth, segment_lengths, segment_count);
#elif NK_TARGET_V128
    return nk_attention_pack_size_i8_v128(key_value_head_count, depth, segment_lengths, segment_count);
#else
    return nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
#endif
}

NK_API_COMPTIME void nk_attention_packed_shape_i8(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                  nk_size_t *segments) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_packed_shape_i8_sapphireamx(key_value_packed, heads, depth, segments);
#elif NK_TARGET_ICELAKE
    nk_attention_packed_shape_i8_icelake(key_value_packed, heads, depth, segments);
#elif NK_TARGET_HASWELL
    nk_attention_packed_shape_i8_haswell(key_value_packed, heads, depth, segments);
#elif NK_TARGET_SME
    nk_attention_packed_shape_i8_sme(key_value_packed, heads, depth, segments);
#elif NK_TARGET_NEONSDOT
    nk_attention_packed_shape_i8_neonsdot(key_value_packed, heads, depth, segments);
#elif NK_TARGET_RVV
    nk_attention_packed_shape_i8_rvv(key_value_packed, heads, depth, segments);
#elif NK_TARGET_V128
    nk_attention_packed_shape_i8_v128(key_value_packed, heads, depth, segments);
#else
    nk_attention_packed_shape_i8_serial(key_value_packed, heads, depth, segments);
#endif
}

NK_API_COMPTIME void nk_attention_pack_i8(nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count,
                                          nk_size_t depth, nk_u32_t const *segment_offsets,
                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                          nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                          void *key_value_packed, nk_size_t task_begin, nk_size_t task_end) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_pack_i8_sapphireamx(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                     segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                     task_end);
#elif NK_TARGET_ICELAKE
    nk_attention_pack_i8_icelake(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                 segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                 task_end);
#elif NK_TARGET_HASWELL
    nk_attention_pack_i8_haswell(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                 segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                 task_end);
#elif NK_TARGET_SME
    nk_attention_pack_i8_sme(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                             key_stride_bytes, value_stride_bytes, key_value_packed, task_begin, task_end);
#elif NK_TARGET_NEONSDOT
    nk_attention_pack_i8_neonsdot(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                  segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                  task_end);
#elif NK_TARGET_RVV
    nk_attention_pack_i8_rvv(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                             key_stride_bytes, value_stride_bytes, key_value_packed, task_begin, task_end);
#elif NK_TARGET_V128
    nk_attention_pack_i8_v128(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                              segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                              task_end);
#else
    nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                task_end);
#endif
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_i8(nk_i8_t const *queries, void const *key_value_packed,
                                                          nk_f32_t *output, nk_size_t head_count,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                          nk_size_t output_stride_bytes, nk_f32_t scale,
                                                          nk_size_t task_start, nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_bidirectional_packed_i8_sapphireamx(queries, key_value_packed, output, head_count,
                                                     key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                     output_stride_bytes, scale, task_start, task_count);
#elif NK_TARGET_ICELAKE
    nk_attention_bidirectional_packed_i8_icelake(queries, key_value_packed, output, head_count, key_value_head_count,
                                                 depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                 task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_bidirectional_packed_i8_haswell(queries, key_value_packed, output, head_count, key_value_head_count,
                                                 depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                 task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_bidirectional_packed_i8_sme(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                             query_offsets, query_stride_bytes, output_stride_bytes, scale, task_start,
                                             task_count);
#elif NK_TARGET_NEONSDOT
    nk_attention_bidirectional_packed_i8_neonsdot(queries, key_value_packed, output, head_count, key_value_head_count,
                                                  depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                  task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_bidirectional_packed_i8_rvv(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                             query_offsets, query_stride_bytes, output_stride_bytes, scale, task_start,
                                             task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_bidirectional_packed_i8_v128relaxed(queries, key_value_packed, output, head_count,
                                                     key_value_head_count, depth, query_offsets, query_stride_bytes,
                                                     output_stride_bytes, scale, task_start, task_count);
#else
    nk_attention_bidirectional_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count,
                                                depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                                task_start, task_count);
#endif
}

NK_API_COMPTIME void nk_attention_causal_packed_i8(nk_i8_t const *queries, void const *key_value_packed,
                                                   nk_f32_t *output, nk_size_t head_count,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                   nk_size_t output_stride_bytes, nk_f32_t scale,
                                                   nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
                                                   nk_size_t task_count) {
#if NK_TARGET_SAPPHIREAMX
    nk_attention_causal_packed_i8_sapphireamx(queries, key_value_packed, output, head_count, key_value_head_count,
                                              depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                              diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_ICELAKE
    nk_attention_causal_packed_i8_icelake(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                          query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                          diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_HASWELL
    nk_attention_causal_packed_i8_haswell(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                          query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                          diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_SME
    nk_attention_causal_packed_i8_sme(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                      window, task_start, task_count);
#elif NK_TARGET_NEONSDOT
    nk_attention_causal_packed_i8_neonsdot(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                           query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                           diagonal_offset, window, task_start, task_count);
#elif NK_TARGET_RVV
    nk_attention_causal_packed_i8_rvv(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                      window, task_start, task_count);
#elif NK_TARGET_V128RELAXED
    nk_attention_causal_packed_i8_v128relaxed(queries, key_value_packed, output, head_count, key_value_head_count,
                                              depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                              diagonal_offset, window, task_start, task_count);
#else
    nk_attention_causal_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                         query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                         window, task_start, task_count);
#endif
}

#endif // !NK_RUNTIME_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_ATTENTION_H
