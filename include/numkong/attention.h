/**
 *  @file include/numkong/attention.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *  @brief SIMD-accelerated ragged Transformer attention.
 *
 *  Contains the following dispatch points, each with a size/pack/compute triple:
 *
 *  - `nk_attention_pack_size_<dtype>_best` - bytes needed to pack a ragged K/V batch
 *  - `nk_attention_pack_<dtype>_best` - one-time K/V packing into a backend-opaque layout
 *  - `nk_attention_bidirectional_packed_<dtype>_best` - bidirectional attention over the batch
 *  - `nk_attention_causal_packed_<dtype>_best` - causal, optionally windowed attention
 *
 *  For dtypes:
 *
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *  - 8-bit @c e4m3 floating point numbers → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - x86: Haswell, Skylake, Genoa, Sapphire Rapids AMX
 *  - Arm: SME, and per-feature NEON capabilities (BFDOT for BF16, FHM for E4M3, SDOT for I8)
 *  - NVIDIA: the CUDA baseline on every device, Ampere @c mma.sync from compute capability 8.0,
 *    Hopper @c wgmma, and native E4M3 on datacenter Blackwell and Blackwell RTX
 *  - AMD: the ROCm baseline on every device, CDNA4 and CDNA5 matrix cores
 *  - portable serial fallback
 *
 *  @section attention_usage Usage and Benefits
 *
 *  Transformer inference packs many variable-length segments into one flat token buffer, and this
 *  family computes attention for the whole batch in one call over UForm's @c segment_offsets and
 *  @c segment_lengths prefix sums, the cu_seqlens convention. Self-attention, cross-attention, and
 *  single-query pooling share one kernel and differ only in their per-segment query counts:
 *
 *  @code{.c}
 *  nk_u32_t offsets[] = {0, 100, 630, 663}, lengths[] = {100, 530, 33};   // 3 ragged segments
 *  nk_capability_t cpu;
 *  nk_size_t bytes;
 *  nk_cpu_capabilities_enabled(&cpu);
 *  nk_attention_pack_size_bf16_best(kv_heads, 128, lengths, 3, cpu, &bytes);
 *  void *kv = aligned_alloc(64, bytes);
 *  nk_attention_pack_bf16_best(keys, values, kv_heads, 128, offsets, lengths, 3, stride, stride, kv, 0, 3 * kv_heads,
 *                              cpu, NULL);
 *  nk_attention_bidirectional_packed_bf16_best(queries, kv, out, heads, kv_heads, 128, offsets, stride, out_stride,
 *                                              scale, 0, NUMKONG_SIZE_MAX, cpu, NULL);
 *  nk_u32_t one_each[] = {0, 1, 2, 3};  // attention pool: 1 learned query per segment
 *  nk_attention_bidirectional_packed_bf16_best(pool_q, kv, pooled, heads, kv_heads, 128, one_each, stride,
 *                                              out_stride, scale, 0, NUMKONG_SIZE_MAX, cpu, NULL);
 *  @endcode
 *
 *  Q, K, V, O use the activations-natural layout of shape @b [tokens,heads,depth] with byte
 *  strides, so a fused QKV projection output of shape @b [tokens,3,hidden] is consumable in place.
 *  Packing takes a [task_begin, task_end) window over the flat @b [segments,kv_heads] grid, and
 *  attention a [task_start, task_start + task_count) window over the flat @b [segments,heads] grid;
 *  tasks touch disjoint outputs, so callers parallelize by distributing tasks across threads — one
 *  per physical core, longest segments first. Outputs are F32: every consumer in a transformer
 *  block — normalization, residual epilogues — wants the accumulator precision anyway.
 *
 *  Unlike cuDNN's fused attention — head dimensions ≤ 256 and a multiple of 8 for 16-bit dtypes —
 *  any `depth ≥ 1` is supported: SIMD backends cover 1…256 with internal zero-padding, and larger
 *  head dimensions transparently fall back to the width-agnostic serial kernel — the fallback rule
 *  is a pure function of the arguments, so packing and attention always agree on the buffer format.
 *  Causal masking lives in `nk_attention_causal_packed_<dtype>_best`, a separate symbol over the
 *  same pack, rather than a flag on the bidirectional kernel.
 *
 *  @section attention_research Open Research Directions
 *
 *  Score-function replacements. On AMX, tile registers support only load/store/zero and
 *  matrix-multiply — no elementwise ops — so any per-pair scoring function — softmax, sigmoid,
 *  ReLU² — forces the O(n²) score matrix through one memory → vector → memory round trip per panel.
 *  Measured on one Sapphire Rapids core at q = kv = 1024, d = 128: softmax ≈ 0.82 TFLOPS, ReLU² ≈
 *  1.1 TFLOPS, sigmoid ≈ softmax — the division costs what the max/sum bookkeeping saves — and the
 *  tile ops alone ≈ 3.1 TFLOPS. ReLU²-scored attention is the cheapest per-pair option, but has no
 *  known production deployments and requires training-time adoption with QK-norm, LayerScale, and
 *  1/n scaling; validation loss does not predict its downstream failures, so retrieval-style probes
 *  are the gate. No zero-shot, quantization-style, softmax → ReLU² conversion exists; the nearest
 *  published path is a short annealing phase at the end of pretraining.
 *
 *  @see ReLU/n scoring at softmax parity in ViTs: https://arxiv.org/abs/2309.08586
 *  @see Sigmoid attention theory, with ReLU² baselines and the QK-norm plus LayerScale stack: https://arxiv.org/abs/2409.04431
 *  @see 1-3B replication of 20 modifications, showing the hidden sigmoid retrieval collapse: https://arxiv.org/abs/2605.20798
 *  @see Polynomial softmax substitutes as Frobenius-norm regularization of attention: https://arxiv.org/abs/2410.18613
 *
 *  Linearized attention, also called kernelized, computes O = φ(Q) · (φ(K)ᵀ V), moving the
 *  nonlinearity from per-pair to per-token: the O(n · d) feature maps run on vector units while
 *  both contractions stay in the matrix unit, meeting at a d × d intermediate — the only attention
 *  class with no O(n²) tile ↔ vector crossover at all, and O(n · d²) complexity — ≈128× less
 *  arithmetic at 16K tokens, d = 128. Bottlenecks: quality at contrastive encoder scale is
 *  unproven; production encoder adoption is near zero; converting pretrained softmax checkpoints
 *  needs distillation — 0.005-2% of pretraining tokens — never a gradient-free swap; and the d × d
 *  state must requantize to BF16 between the two matrix multiplications.
 *
 *  @see EfficientViT-SAM, a shipped ReLU-kernel linear attention encoder at SAM-ViT-H quality: https://arxiv.org/abs/2402.05008
 *  @see LoLCATs low-rank linearization of Llamas: https://arxiv.org/abs/2410.10254
 *  @see RADLADS conversion at <0.005% of pretraining: https://arxiv.org/abs/2505.03005
 *  @see Hedgehog, why zero-shot kernel swaps collapse: https://arxiv.org/abs/2402.04347
 *
 *  @section attention_causal Causal and Sliding-Window Attention
 *
 *  `nk_attention_causal_packed_*` covers decoder inference with two scalars: query row @c r sits at
 *  position `p = r + diagonal_offset` and sees the @c window keys ending at @c p, inclusive.
 *  `offset = 0` gives causal prefill, `offset = position_count − row_count` gives decode and
 *  chunked prefill against a longer cache, a finite @c window gives sliding-window attention, and
 *  the ragged segment directory already provides block-diagonal document masking for packed
 *  batches. Rows whose range is empty, such as `p < 0` or `window = 0`, produce zeros, as do
 *  segments with no keys in both modes.
 *
 *  Masking clips each row's key range instead of writing −∞ scores, because the clamped @c exp2,
 *  SME rounding, and I8 score differences all misbehave on sentinels. KV panels outside every row
 *  of a query block are skipped outright, so causality is ≈2× fewer FLOPs at equal context, and a
 *  row that sees no key of a panel keeps its running maximum. Deliberately out of scope: ALiBi,
 *  logit soft-capping, and paged KV caches.
 *
 *  @see FlashAttention-2 causal tiling and work skipping: https://arxiv.org/abs/2307.08691
 *  @see Mistral 7B, sliding-window attention in production: https://arxiv.org/abs/2310.06825
 *
 *  @section attention_references References
 *
 *  @see FlashAttention-2 tiling and the online softmax: https://arxiv.org/abs/2307.08691
 *  @see cuDNN attention shape constraints for comparison: https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef NUMKONG_ATTENTION_H
#define NUMKONG_ATTENTION_H

#include "numkong/types.h"
#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Returns the packed KV-cache size in bytes for a ragged batch of segments.
 *  @param[in] key_value_head_count Number of K/V heads, a nonzero divisor of the query head count;
 *      attention must take it and @p depth as packed, and only debug builds assert these rules.
 *  @param[in] depth Head dimension; any value ≥ 1.
 *  @param[in] segment_lengths Live token counts, one per segment; zeros allowed.
 *  @param[in] segment_count Number of segments packed together.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the packed buffer.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                       nk_capability_t capabilities, nk_size_t *bytes);

/**
 *  @brief Reads a packed KV cache's shape from its header.
 *  @param[in] key_value_packed A buffer produced by the matching @c nk_attention_pack_bf16_best.
 *  @param[out] heads Receives the K/V head count.
 *  @param[out] depth Receives the head dimension.
 *  @param[out] segments Receives the segment count.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_best(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments,
                                                          nk_capability_t capabilities, void *stream);

/**
 *  @brief Packs a ragged batch of K and V segments into a backend-opaque layout.
 *  @param[in] keys,values Token-major matrices, one row of @p key_value_head_count × @p depth
 *      elements per token, with strided rows.
 *  @param[in] segment_offsets Start token of each segment, @p segment_count + 1 prefix sums.
 *  @param[in] segment_lengths Live token counts, one per segment; zeros mark padding slots.
 *  @param[in] key_stride_bytes Row (token) stride of @p keys in bytes.
 *  @param[in] value_stride_bytes Row (token) stride of @p values in bytes.
 *  @param[out] key_value_packed 64-byte-aligned buffer of `nk_attention_pack_size_*` bytes.
 *  @param[in] task_begin First task of a window over the segments × K/V heads grid.
 *  @param[in] task_end End of that half-open window, clipped to the grid.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  Windows let callers pack in parallel: tasks write disjoint ranges, and the header and directory
 *  are written by the window starting at task 0.
 */
NUMKONG_API nk_status_t nk_attention_pack_bf16_best(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_best(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_best(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end,
                                                  nk_capability_t capabilities, void *stream);

/**
 *  @brief Ragged bidirectional scaled-dot-product attention.
 *
 *  Computes O[s] = softmax(Q[s] × K[s]ᵀ × scale) × V[s] for every segment s. Covers self-attention,
 *  where @p query_offsets equal the pack-time @c segment_offsets, cross-attention, and pooling,
 *  where `query_offsets = {0, 1, 2, …}` holds one query per segment, plus GQA/MQA via
 *  @p key_value_head_count < @p head_count. Pack @p key_value_packed with the same
 *  @p capabilities: CPU kernels return @c nk_pack_mismatch_k for buffers another capability packed,
 *  while GPU kernels trust it, as reading its device header would wait on the stream.
 *
 *  @param[in] queries Token-major matrix, one row of @p head_count × @p depth elements per query
 *      token, with @p query_stride_bytes bytes between rows.
 *  @param[in] key_value_packed Buffer produced by the matching `nk_attention_pack_*` backend.
 *  @param[out] output Token-major F32 matrix, one row of @p head_count × @p depth elements per
 *      query token, with @p output_stride_bytes bytes between rows.
 *  @param[in] query_offsets First query row of each segment, as segment count + 1 prefix sums.
 *  @param[in] scale Score multiplier, typically 1 / √depth.
 *  @param[in] task_start,task_count Window over the segments × heads grid, with @p task_count
 *      clipped to segments × heads − @p task_start. Tasks write disjoint output regions, so
 *      callers parallelize freely.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_best(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream);

/**
 *  @brief Ragged causal scaled-dot-product attention with an optional sliding window.
 *
 *  Query row @c r of a segment sits at position `p = r + diagonal_offset` and attends to the keys
 *  `[max(0, p − window + 1), min(p, length − 1)]`; rows with an empty range produce zeros.
 *  Packing and every other parameter follow @c nk_attention_bidirectional_packed_bf16_best.
 *
 *  @param[in] diagonal_offset Position of query row 0: `0` for prefill, `length − query_count`
 *      against a cache.
 *  @param[in] window Visible keys including the query itself; @c NUMKONG_SIZE_MAX is unbounded, `0`
 *      masks every key.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_best(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_best(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_best(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_best(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_best(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream);

/**
 *  @brief NeoX split-half rotary position embedding (RoPE): rotates channel pairs
 *      by per-token angles.
 *
 *  Rotates the channel pair i and i + h of every head in every row, h being half the @p depth:
 *
 *  @verbatim
 *  y[i]     = x[i] · cos - x[i + h] · sin
 *  y[i + h] = x[i] · sin + x[i + h] · cos
 *  @endverbatim
 *
 *  @param[in] x Input token matrix of shape rows by @p head_count × @p depth.
 *  @param[in] cos Per-token cosine angle grid of shape rows by @p depth / 2, shared across heads.
 *  @param[in] sin Per-token sine angle grid of shape rows by @p depth / 2, shared across heads.
 *  @param[out] y Output matrix, same shape and dtype as x; may alias x for in-place rotation.
 *  @param[in] rows The number of token rows.
 *  @param[in] head_count The number of heads per token.
 *  @param[in] depth The even number of channels per head.
 *  @param[in] x_stride_bytes Row (token) stride of x in bytes.
 *  @param[in] y_stride_bytes Row (token) stride of y in bytes.
 *  @param[in] input_scale Scalar folded onto every loaded element (E4M3 descale; 1.0 for BF16/F32).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_rope_f32_best(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                   nk_f32_t input_scale, nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_best(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, nk_capability_t capabilities, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, nk_capability_t capabilities, void *stream);

/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_serial(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_serial(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_serial(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_serial(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_serial(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_serial(nk_i8_t const *keys, nk_i8_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_serial(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_serial(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_serial(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_serial(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_serial(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_serial(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_serial(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_f32_t *,
                                                     nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                     void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_serial(nk_bf16_t const *, nk_f32_t const *, nk_f32_t const *,
                                                      nk_bf16_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                      nk_size_t, nk_f32_t, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_serial(nk_e4m3_t const *, nk_f32_t const *, nk_f32_t const *,
                                                      nk_e4m3_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                      nk_size_t, nk_f32_t, void *stream);

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_haswell(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_haswell(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_haswell(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_haswell(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_haswell(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_haswell(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_haswell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_haswell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_haswell(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_haswell(nk_i8_t const *keys, nk_i8_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_haswell(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_haswell(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_haswell(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_f32_t *,
                                                      nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                      void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_haswell(nk_bf16_t const *, nk_f32_t const *, nk_f32_t const *,
                                                       nk_bf16_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                       nk_size_t, nk_f32_t, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_haswell(nk_e4m3_t const *, nk_f32_t const *, nk_f32_t const *,
                                                       nk_e4m3_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                       nk_size_t, nk_f32_t, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_skylake(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_skylake(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_skylake(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_skylake(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_skylake(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_skylake(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_skylake(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_skylake(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_skylake(nk_f32_t const *, nk_f32_t const *, nk_f32_t const *, nk_f32_t *,
                                                      nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_size_t, nk_f32_t,
                                                      void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_skylake(nk_bf16_t const *, nk_f32_t const *, nk_f32_t const *,
                                                       nk_bf16_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                       nk_size_t, nk_f32_t, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_skylake(nk_e4m3_t const *, nk_f32_t const *, nk_f32_t const *,
                                                       nk_e4m3_t *, nk_size_t, nk_size_t, nk_size_t, nk_size_t,
                                                       nk_size_t, nk_f32_t, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_icelake(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_icelake(nk_i8_t const *keys, nk_i8_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_icelake(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_icelake(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_genoa(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_genoa(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_genoa(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_genoa(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_genoa(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_genoa(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_genoa(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_genoa(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_sapphireamx(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_offsets,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                           void *key_value_packed, nk_size_t task_begin,
                                                           nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_sapphireamx(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_sapphireamx(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_sapphireamx(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_offsets,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                           void *key_value_packed, nk_size_t task_begin,
                                                           nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_sapphireamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_sapphireamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_sapphireamx(void const *key_value_packed, nk_size_t *heads,
                                                                 nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_sapphireamx(nk_i8_t const *keys, nk_i8_t const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_offsets,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                         void *key_value_packed, nk_size_t task_begin,
                                                         nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_sapphireamx(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_sapphireamx(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_SAPPHIREAMX

/*  Diamond Rapids AMX provides only the E4M3 attention variant: its native FP8 tiles, driven by
 *  @c _tile_dphf8ps, are its differentiator, while its I8/BF16 paths would merely clone the
 *  Sapphire AMX backend. */
#if NUMKONG_TARGET_DIAMONDAMX
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                               nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_diamondamx(void const *key_value_packed, nk_size_t *heads,
                                                                  nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_diamondamx(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_offsets,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                          void *key_value_packed, nk_size_t task_begin,
                                                          nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_diamondamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_diamondamx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_DIAMONDAMX

#if NUMKONG_TARGET_SME
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_sme(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_sme(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_sme(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_sme(nk_bf16_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_sme(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_sme(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_sme(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_sme(nk_e4m3_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                      nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_sme(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_sme(nk_i8_t const *keys, nk_i8_t const *values,
                                                 nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                 nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                 nk_size_t value_stride_bytes, void *key_value_packed,
                                                 nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_sme(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_sme(nk_i8_t const *queries, void const *key_value_packed,
                                                          nk_f32_t *output, nk_size_t head_count,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                          nk_size_t output_stride_bytes, nk_f32_t scale,
                                                          nk_i64_t diagonal_offset, nk_size_t window,
                                                          nk_size_t task_start, nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_neonbfdot(void const *key_value_packed, nk_size_t *heads,
                                                                 nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_neonbfdot(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_offsets,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                         void *key_value_packed, nk_size_t task_begin,
                                                         nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_neonbfdot(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_neonbfdot(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_neonfhm(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_neonfhm(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                       nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                       nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                       nk_size_t value_stride_bytes, void *key_value_packed,
                                                       nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_neonfhm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_neonfhm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_neonsdot(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_neonsdot(nk_i8_t const *keys, nk_i8_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_neonsdot(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_neonsdot(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_RVV
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_rvv(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_rvv(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                      nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_rvv(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_rvv(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_rvv(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_rvv(nk_i8_t const *keys, nk_i8_t const *values,
                                                 nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                 nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                 nk_size_t value_stride_bytes, void *key_value_packed,
                                                 nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_rvv(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_rvv(nk_bf16_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_rvv(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_rvv(nk_e4m3_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_rvv(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_rvv(nk_i8_t const *queries, void const *key_value_packed,
                                                          nk_f32_t *output, nk_size_t head_count,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                          nk_size_t output_stride_bytes, nk_f32_t scale,
                                                          nk_i64_t diagonal_offset, nk_size_t window,
                                                          nk_size_t task_start, nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                 nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_v128relaxed(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_offsets,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                           void *key_value_packed, nk_size_t task_begin,
                                                           nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_v128relaxed(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_offsets,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                           void *key_value_packed, nk_size_t task_begin,
                                                           nk_size_t task_end, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_v128relaxed(nk_i8_t const *keys, nk_i8_t const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_offsets,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                         void *key_value_packed, nk_size_t task_begin,
                                                         nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_v128relaxed(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_v128relaxed(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_v128relaxed(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_v128relaxed(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_v128relaxed(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_v128relaxed(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

/*  GPU kernels take their CPU counterparts' arguments and return without waiting on the device.
 *  Only @c pack_size reads @c segment_lengths on the host, so every other pointer must be device
 *  or managed memory. Apple GPUs have no attention kernels.
 *
 *  CUDA on every NVIDIA device: 32 lanes per query row, two sweeps over its keys as the serial
 *  backend makes, and the pack routine every GPU capability shares. */
#if NUMKONG_TARGET_CUDA
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cuda(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cuda(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_cuda(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_cuda(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cuda(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cuda(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_cuda(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_cuda(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cuda(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cuda(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_cuda(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_cuda(nk_i8_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_cuda(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                   nk_f32_t input_scale, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_cuda(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, void *stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_cuda(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, void *stream);
#endif // NUMKONG_TARGET_CUDA

/*  NVIDIA backends from Ampere on: FlashAttention-2 on warp-level `mma.sync` up to depth 256, the
 *  CUDA baseline past it. */
#if NUMKONG_TARGET_AMPERE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_ampere(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_ampere(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_ampere(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_ampere(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_ampere(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_ampere(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_ampere(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_ampere(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_ampere(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_ampere(nk_i8_t const *keys, nk_i8_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_ampere(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_ampere(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_AMPERE

/*  NVIDIA Hopper backends, compute capability 9.0, through warpgroup @c wgmma over shared-memory
 *  descriptors. */
#if NUMKONG_TARGET_HOPPER
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_hopper(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_hopper(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_hopper(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_hopper(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_hopper(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_hopper(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                      nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                      nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                      nk_size_t value_stride_bytes, void *key_value_packed,
                                                      nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_hopper(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_hopper(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_hopper(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_hopper(nk_i8_t const *keys, nk_i8_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_hopper(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_hopper(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_HOPPER

/*  NVIDIA datacenter Blackwell backends, the compute capability 10.x family, through single-thread
 *  @c tcgen05 MMAs into tensor memory. */
#if NUMKONG_TARGET_BLACKWELL
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_blackwell(void const *key_value_packed, nk_size_t *heads,
                                                                 nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_blackwell(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                         nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_offsets,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                         void *key_value_packed, nk_size_t task_begin,
                                                         nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_blackwell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_blackwell(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_BLACKWELL

/*  NVIDIA backends for the compute capability 12.x family, with E4M3 on the tensor cores natively.
 *  BF16 and I8 there use the Ampere kernels, which already run at the native rate. */
#if NUMKONG_TARGET_BLACKWELLRTX
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_blackwellrtx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_u32_t const *segment_lengths,
                                                                 nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_blackwellrtx(void const *key_value_packed, nk_size_t *heads,
                                                                    nk_size_t *depth, nk_size_t *segments,
                                                                    void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_blackwellrtx(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *segment_offsets,
                                                            nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                            nk_size_t key_stride_bytes, nk_size_t value_stride_bytes,
                                                            void *key_value_packed, nk_size_t task_begin,
                                                            nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_blackwellrtx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_blackwellrtx(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_BLACKWELLRTX

/*  ROCm on every AMD device: the CUDA baseline's source, compiled by HIP. */
#if NUMKONG_TARGET_ROCM
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_rocm(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_rocm(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_rocm(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_rocm(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_rocm(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_rocm(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_rocm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_rocm(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_rocm(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_rocm(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_rocm(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_rocm(nk_i8_t const *queries, void const *key_value_packed,
                                                           nk_f32_t *output, nk_size_t head_count,
                                                           nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                           nk_size_t output_stride_bytes, nk_f32_t scale,
                                                           nk_i64_t diagonal_offset, nk_size_t window,
                                                           nk_size_t task_start, nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_ROCM

/*  AMD Instinct MI350 backends, gfx950, through its matrix cores. */
#if NUMKONG_TARGET_CDNA4
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cdna4(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cdna4(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_cdna4(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_cdna4(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cdna4(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cdna4(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_cdna4(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_cdna4(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cdna4(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cdna4(nk_i8_t const *keys, nk_i8_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_cdna4(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_cdna4(nk_i8_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_CDNA4

/*  AMD Instinct MI400 backends, through its matrix cores. */
#if NUMKONG_TARGET_CDNA5
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cdna5(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cdna5(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_cdna5(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_cdna5(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cdna5(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cdna5(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                     nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                     nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                     nk_size_t value_stride_bytes, void *key_value_packed,
                                                     nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_cdna5(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_e4m3_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_cdna5(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, void *stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cdna5(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments, void *stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cdna5(nk_i8_t const *keys, nk_i8_t const *values,
                                                   nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                   nk_size_t value_stride_bytes, void *key_value_packed,
                                                   nk_size_t task_begin, nk_size_t task_end, void *stream);
/** @copydoc nk_attention_bidirectional_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_cdna5(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count, void *stream);
/** @copydoc nk_attention_causal_packed_i8_best */
NUMKONG_API nk_status_t nk_attention_causal_packed_i8_cdna5(nk_i8_t const *queries, void const *key_value_packed,
                                                            nk_f32_t *output, nk_size_t head_count,
                                                            nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
                                                            nk_size_t output_stride_bytes, nk_f32_t scale,
                                                            nk_i64_t diagonal_offset, nk_size_t window,
                                                            nk_size_t task_start, nk_size_t task_count, void *stream);
#endif // NUMKONG_TARGET_CDNA5

/** Returns the output dtype for attention: accumulator-precision F32 for all inputs. */
NUMKONG_INLINE nk_dtype_t nk_attention_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_bf16_k: return nk_f32_k;
    case nk_e4m3_k: return nk_f32_k;
    case nk_i8_k: return nk_f32_k; // quantized-weight softmax, F32 outputs
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of RoPE outputs before they round into their type, relative to the two
 *  products each of them sums: 3 roundings in F32, which every input type rotates in, through the
 *  descale, the product and the sum. */
NUMKONG_INLINE nk_f64_t nk_attention_rope_error_bound(nk_dtype_t dtype) {
    nk_unused_(dtype);
    return 3 * nk_accumulation_error_bound(nk_f32_k);
}

/**
 *  @brief Finds the attention kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_attention_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                 nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
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
#include "numkong/attention/simt.cuh"
#include "numkong/attention/ampere.cuh"
#include "numkong/attention/hopper.cuh"
#include "numkong/attention/blackwell.cuh"
#include "numkong/attention/blackwellrtx.cuh"
#include "numkong/attention/cdna4.cuh"
#include "numkong/attention/cdna5.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(key_value_packed), nk_unused_(heads), nk_unused_(depth), nk_unused_(segments), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_best(nk_bf16_t const *keys, nk_bf16_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end,
                                                    nk_capability_t capabilities, void *stream) {
    nk_unused_(keys), nk_unused_(values), nk_unused_(key_value_head_count), nk_unused_(depth),
        nk_unused_(segment_offsets), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(key_stride_bytes), nk_unused_(value_stride_bytes), nk_unused_(key_value_packed),
        nk_unused_(task_begin), nk_unused_(task_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_bidirectional_packed_bf16_best(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(task_start), nk_unused_(task_count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_causal_packed_bf16_best(
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(diagonal_offset), nk_unused_(window),
        nk_unused_(task_start), nk_unused_(task_count), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_best(void const *key_value_packed, nk_size_t *heads,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(key_value_packed), nk_unused_(heads), nk_unused_(depth), nk_unused_(segments), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_best(nk_e4m3_t const *keys, nk_e4m3_t const *values,
                                                    nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                    nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                    nk_size_t value_stride_bytes, void *key_value_packed,
                                                    nk_size_t task_begin, nk_size_t task_end,
                                                    nk_capability_t capabilities, void *stream) {
    nk_unused_(keys), nk_unused_(values), nk_unused_(key_value_head_count), nk_unused_(depth),
        nk_unused_(segment_offsets), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(key_stride_bytes), nk_unused_(value_stride_bytes), nk_unused_(key_value_packed),
        nk_unused_(task_begin), nk_unused_(task_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_bidirectional_packed_e4m3_best(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(task_start), nk_unused_(task_count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_causal_packed_e4m3_best(
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(diagonal_offset), nk_unused_(window),
        nk_unused_(task_start), nk_unused_(task_count), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                       nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_best(void const *key_value_packed, nk_size_t *heads,
                                                          nk_size_t *depth, nk_size_t *segments,
                                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(key_value_packed), nk_unused_(heads), nk_unused_(depth), nk_unused_(segments), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_best(nk_i8_t const *keys, nk_i8_t const *values,
                                                  nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,
                                                  nk_size_t segment_count, nk_size_t key_stride_bytes,
                                                  nk_size_t value_stride_bytes, void *key_value_packed,
                                                  nk_size_t task_begin, nk_size_t task_end,
                                                  nk_capability_t capabilities, void *stream) {
    nk_unused_(keys), nk_unused_(values), nk_unused_(key_value_head_count), nk_unused_(depth),
        nk_unused_(segment_offsets), nk_unused_(segment_lengths), nk_unused_(segment_count),
        nk_unused_(key_stride_bytes), nk_unused_(value_stride_bytes), nk_unused_(key_value_packed),
        nk_unused_(task_begin), nk_unused_(task_end), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_bidirectional_packed_i8_best(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,
    nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(task_start), nk_unused_(task_count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_causal_packed_i8_best(
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride_bytes,
    nk_size_t output_stride_bytes, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start,
    nk_size_t task_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(queries), nk_unused_(key_value_packed), nk_unused_(output), nk_unused_(head_count),
        nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets), nk_unused_(query_stride_bytes),
        nk_unused_(output_stride_bytes), nk_unused_(scale), nk_unused_(diagonal_offset), nk_unused_(window),
        nk_unused_(task_start), nk_unused_(task_count), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_best(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                   nk_f32_t input_scale, nk_capability_t capabilities, void *stream) {
    nk_unused_(x), nk_unused_(y), nk_unused_(cos), nk_unused_(sin), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride_bytes), nk_unused_(y_stride_bytes), nk_unused_(input_scale),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_best(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, nk_capability_t capabilities, void *stream) {
    nk_unused_(x), nk_unused_(y), nk_unused_(cos), nk_unused_(sin), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride_bytes), nk_unused_(y_stride_bytes), nk_unused_(input_scale),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride_bytes, nk_size_t y_stride_bytes,
                                                    nk_f32_t input_scale, nk_capability_t capabilities, void *stream) {
    nk_unused_(x), nk_unused_(y), nk_unused_(cos), nk_unused_(sin), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride_bytes), nk_unused_(y_stride_bytes), nk_unused_(input_scale),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                 nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_ATTENTION_H
