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
 *  - `nk_attention_packed_<dtype>_best` - attention over the batch under a band of visible keys
 *  - `nk_attention_packed_gradients_<dtype>_best` - the gradients with respect to queries, keys and
 *    values, for training
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
 *  - AMD: the ROCm baseline on every device, CDNA3, CDNA4 and CDNA5 matrix cores
 *  - Apple: the Metal baseline on every GPU, SIMD-group matrices from family 9, and the tensor
 *    operations of the family 10 Neural Accelerators
 *  - portable serial fallback
 *
 *  @section attention_usage Usage and Benefits
 *
 *  Transformer inference packs many variable-length segments into one flat token buffer, and this
 *  family computes attention for the whole batch in one call over UForm's @c key_offsets prefix
 *  sums, the cu_seqlens convention, and optional @c key_lengths. Self-attention, cross-attention,
 *  and single-query pooling share one kernel and differ only in their per-segment query counts:
 *
 *  @code{.c}
 *  nk_u32_t offsets[] = {0, 100, 630, 663}, lengths[] = {100, 530, 33};   // 3 ragged segments
 *  nk_capability_t cpu;
 *  nk_size_t bytes;
 *  nk_cpu_capabilities_enabled(&cpu);
 *  nk_attention_pack_size_bf16_best(kv_heads, 128, 663, 3, cpu, &bytes);   // 663 tokens in 3 segments
 *  void *kv = aligned_alloc(64, bytes);
 *  nk_size_t const bf16 = sizeof(nk_bf16_t);
 *  nk_attention_pack_bf16_best(kv_heads, 128, offsets, lengths, 3, keys, stride, values, stride, kv, 0,
 *                              3 * kv_heads, cpu, NULL);
 *  nk_size_t const all = NUMKONG_SIZE_MAX, tasks = 663 * heads;   // every query token × head
 *  nk_attention_packed_bf16_best(heads, kv_heads, 128, offsets, 663, scale, all, all, queries, stride, kv, out,
 *                                out_stride, NULL, 0, tasks, cpu, NULL);
 *  nk_u32_t one_each[] = {0, 1, 2, 3}; // attention pool: 1 learned query per segment
 *  nk_attention_packed_bf16_best(heads, kv_heads, 128, one_each, 3, scale, all, all, pool_q, stride, kv, pooled,
 *                                out_stride, NULL, 0, 3 * heads, cpu, NULL);
 *  @endcode
 *
 *  Q, K, V, O use the activations-natural layout of shape @b [tokens,heads,depth] with byte
 *  strides, so a fused QKV projection output of shape @b [tokens,3,hidden] is consumable in place.
 *  Packing takes a [tasks_begin, tasks_end) window over the flat @b [segments,kv_heads] grid, and
 *  attention one over the flat @b [query_tokens,heads] grid of output rows, which @c log_sum_exp
 *  shares. Tasks touch disjoint outputs and windows run in any order, so callers parallelize by
 *  cutting each grid into a few windows of equal cost per thread, one thread per physical core, an
 *  attention row costing the keys it sees. Outputs are F32: every consumer in a transformer block,
 *  such as normalization and residual epilogues, wants the accumulator precision anyway.
 *
 *  Unlike cuDNN's fused attention, limited to head dimensions ≤ 256 and a multiple of 8 for 16-bit
 *  dtypes, any `depth ≥ 1` is supported: SIMD backends cover 1…256 with internal zero-padding, and
 *  larger head dimensions fall back to the width-agnostic serial kernel. The fallback rule is a
 *  pure function of the arguments, so packing and attention always agree on the buffer format.
 *
 *  Attention @c packed is the twin of @c nk_dots_packed: queries meet packed keys and values as A
 *  meets packed B. A family gets a new verb when its operands or outputs differ, and a parameter
 *  when only the visible region does: @c nk_dots_symmetric is a verb, as it reads one operand and
 *  fills the upper triangle, the @c nk_diagonal_band_t (0, NUMKONG_SIZE_MAX), that A × Aᵀ mirrors.
 *  Masking keeps the operands and outputs of @c packed, so it is the @c keys_before and
 *  @c keys_after band of every attention verb.
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
 *  @section attention_masks Causal, Sliding-Window and Bidirectional Masks
 *
 *  Each segment's queries align to the end of its keys: row @c r of a segment with @c q queries and
 *  @c k keys sits at the signed position `p = r + k − q`, and key `j < k` is visible when
 *  `p − keys_before ≤ j ≤ p + keys_after`, with saturating arithmetic and @c NUMKONG_SIZE_MAX
 *  unbounded on either side. Causal attention is (NUMKONG_SIZE_MAX, 0), a sliding window of @c w
 *  keys is (w − 1, 0), and bidirectional attention is (NUMKONG_SIZE_MAX, NUMKONG_SIZE_MAX).
 *  Prefill, decode and ragged chunked prefill against a longer cache need no offset, as every
 *  segment's `k − q` places its rows, and the segment directory already provides block-diagonal
 *  document masking for packed batches. Rows that see no key, such as `p < 0` under a causal mask,
 *  produce zeros and a log-sum-exp of −∞, as do segments with no keys.
 *
 *  Masking clips each row's key range instead of writing −∞ scores, because the clamped @c exp2,
 *  SME rounding, and I8 score differences all misbehave on sentinels. Tiles of a query block and a
 *  KV panel outside the band are skipped outright, so causality is ≈2× fewer FLOPs at equal
 *  context, tiles inside it run unmasked, and a row that sees no key of a panel keeps its running
 *  maximum. Deliberately out of scope: ALiBi, logit soft-capping, and paged KV caches.
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
 *  @brief Returns bytes enough to pack a ragged batch of segments however its tokens split.
 *  @param[in] key_value_head_count Number of K/V heads, a nonzero divisor of the query head count;
 *      attention must take it and @p depth as packed, and only debug builds assert these rules.
 *  @param[in] depth Head dimension; any value ≥ 1.
 *  @param[in] token_count Tokens across all segments, the sum of the pack's key counts.
 *  @param[in] segment_count Number of segments packed together.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[out] bytes The size of the packed buffer.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *  @note The packed layout is backend-specific and must be produced by the matching pack function.
 */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
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
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_best(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_capability_t capabilities,
                                                            nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_best(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_capability_t capabilities,
                                                           nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_best(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_capability_t capabilities,
                                                            nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_best(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments,
                                                          nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_best(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_capability_t capabilities,
                                                             nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_best(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_capability_t capabilities,
                                                             nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream);

/**
 *  @brief Locates a pack's key slot boundaries and key counts.
 *  @param[in] key_value_packed A buffer produced by the matching @c nk_attention_pack_bf16_best.
 *  @param[in] segment_count The pack's segment count, from @c nk_attention_packed_shape_bf16_best.
 *  @param[out] key_offsets Points at the pack's @p segment_count + 1 slot boundaries.
 *  @param[out] key_lengths Points at the pack's @p segment_count key counts.
 *  @return @c nk_success_k.
 *  @note Pure address arithmetic that never dereferences the pack, so it works on device packs too.
 */
NUMKONG_API nk_status_t nk_attention_packed_segments(void const *key_value_packed, nk_size_t segment_count,
                                                     nk_u32_t const **key_offsets, nk_u32_t const **key_lengths);

/**
 *  @brief Packs a ragged batch of K and V segments into a backend-opaque layout.
 *  @param[in] key_offsets The @p segment_count + 1 slot boundaries: segment @c s keeps its keys in
 *      rows `key_offsets[s]` onward, and may leave spare capacity before `key_offsets[s + 1]`.
 *      Must not decrease.
 *  @param[in] key_lengths Keys each segment holds, at most its slot's width; zeros mark padding
 *      slots. Null means every slot is full: the difference of consecutive @p key_offsets.
 *  @param[in] keys,values Token-major matrices, one row of @p key_value_head_count × @p depth
 *      contiguous elements per token, with strided rows. Packing fuses no transposition, so
 *      depth-major K or V is transposed in a separate pass first.
 *  @param[in] key_stride Row (token) stride of @p keys in bytes.
 *  @param[in] value_stride Row (token) stride of @p values in bytes.
 *  @param[out] key_value_packed 64-byte-aligned buffer of `nk_attention_pack_size_*` bytes.
 *  @param[in] tasks_begin First task of a window over the segments × K/V heads grid.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, @c nk_unexpected_dimensions_k for a count over @c UINT32_MAX or
 *      slots that decrease or overflow, or @c nk_missing_kernel_k when no capability in
 *      @p capabilities has it.
 *
 *  GPU packs read the arrays on the device, so only their counts are checked.
 *  Windows let callers pack in parallel and in any order: tasks write disjoint ranges, the window
 *  starting at task 0 also writes the header and directory, and no window reads them.
 */
NUMKONG_API nk_status_t nk_attention_pack_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_bf16_t const *keys,
                                                    nk_size_t key_stride, nk_bf16_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                   nk_f16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_e4m3_t const *keys,
                                                    nk_size_t key_stride, nk_e4m3_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                  nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_capability_t capabilities, nk_stream_t stream);

/** Packs block-scaled K and V as @c nk_attention_pack_bf16_best does, from references to their
 *  codes, block scales and NVFP4 tensor scale; @p depth is a whole number of blocks, and the
 *  strides count bytes of codes. */
NUMKONG_API nk_status_t nk_attention_pack_nvfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream);

/**
 *  @brief Ragged scaled-dot-product attention under a band of visible keys.
 *
 *  Computes O[s] = softmax(Q[s] × K[s]ᵀ × scale) × V[s] for every segment s over the keys each
 *  query row sees, as @ref attention_masks defines. Covers self-attention, where @p query_offsets
 *  equal the pack-time @c key_offsets, cross-attention, and pooling, where
 *  `query_offsets = {0, 1, 2, …}` holds one query per segment, plus GQA/MQA via
 *  @p key_value_head_count < @p head_count. Pack @p key_value_packed with the same
 *  @p capabilities: CPU kernels return @c nk_pack_mismatch_k for buffers another capability packed,
 *  while GPU kernels trust it, as reading its device header would wait on the stream.
 *
 *  @param[in] query_offsets First query row of each segment, as segment count + 1 prefix sums.
 *  @param[in] query_token_count Query rows the call addresses, `query_offsets[segments]`; GPU
 *      kernels size tensor maps and workspaces from it without reading the offsets.
 *  @param[in] scale Score multiplier, typically 1 / √depth.
 *  @param[in] keys_before Keys visible before each query's position: @c NUMKONG_SIZE_MAX for causal
 *      and bidirectional attention, `w − 1` for a sliding window of @c w keys.
 *  @param[in] keys_after Keys visible after each query's position: `0` for causal attention,
 *      @c NUMKONG_SIZE_MAX for bidirectional.
 *  @param[in] queries Token-major matrix, one row of @p head_count × @p depth elements per query
 *      token, with @p query_stride bytes between rows.
 *  @param[in] key_value_packed Buffer produced by the matching `nk_attention_pack_*` backend.
 *  @param[out] output Token-major F32 matrix, one row of @p head_count × @p depth elements per
 *      query token, with @p output_stride bytes between rows.
 *  @param[out] log_sum_exp Optional natural log of Σ exp(score multiplier × q · k) over each row's
 *      keys, one per query token and head, @b [query_tokens,heads] with no padding, −∞ for a row
 *      without keys: what the backward pass recomputes the weights from. Null skips it.
 *  @param[in] tasks_begin First task of a window over the query tokens × heads grid, task
 *      t × @p head_count + h being head @c h of query token @c t, as in @p log_sum_exp.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid's live rows from
 *      @p query_offsets[0] × @p head_count. A window partitions the rows of @p output and
 *      @p log_sum_exp the call writes, per query token × head, so callers parallelize freely; a
 *      window may split a segment or a head group.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_packed_bf16_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);

/**
 *  @brief Gradients of ragged attention with respect to its queries, keys and values.
 *
 *  Recomputes every attention weight from @p log_sum_exp, so nothing of the forward but its output
 *  and that per-row statistic is kept. The gradients are F32 and overwritten: @p query_gradient
 *  holds one row of @p head_count × @p depth elements per query token, while @p key_gradient and
 *  @p value_gradient hold one row of @p key_value_head_count × @p depth elements per key token,
 *  token @c t of segment @c s at row `key_offsets[s] + t`, with the offsets the pack stored. Rows
 *  past a segment's key count, the gaps and tails before the next segment's first key, are never
 *  written, so callers zero them when they matter. Tasks cover the segments × key-value heads grid,
 *  each owning one KV head of one segment together with the query heads sharing it, so tasks write
 *  disjoint gradients and each sums in a fixed order. The Blackwell kernels run a single pass for
 *  BF16 heads of 64 or 128 dimensions with 16-byte aligned rows: it allocates a workspace of
 *  @p query_token_count × @p head_count × (2 × @p depth + 4) bytes with @c cudaMallocAsync on
 *  @p stream and frees it the same way, never synchronizing, and adds the @p query_gradient
 *  partials of different key blocks in F32 in any order, so its low bits may vary run to run. Other
 *  calls, or a failed allocation, take the two-pass path.
 *
 *  @param[in] output The forward's output for these queries and pack.
 *  @param[in] output_gradient Gradient of the loss with respect to @p output, laid out like it.
 *  @param[in] log_sum_exp The forward's natural log of Σ exp(scale × score) over each row's keys,
 *      one per query token and head, @b [query_tokens,heads] with no padding.
 *  @param[out] query_gradient Gradient with respect to @p queries.
 *  @param[in] query_gradient_stride Bytes between the @p query_gradient rows of consecutive tokens.
 *  @param[out] key_gradient Gradient with respect to the packed keys.
 *  @param[out] value_gradient Gradient with respect to the packed values.
 *  @param[in] key_value_gradient_stride Bytes between the gradient rows of consecutive key tokens.
 *  @param[in] keys_before Keys before each query that it saw in the forward pass.
 *  @param[in] keys_after Keys after each query that it saw in the forward pass.
 *  @param[in] tasks_begin First task of a window over the segments × key-value heads grid.
 *  @param[in] tasks_end End of that half-open window, clipped to the grid. A window partitions the
 *      outputs the call writes: segments × K/V heads own their @p key_gradient and
 *      @p value_gradient rows and the @p query_gradient rows of their query heads.
 *
 *  Every other parameter follows @c nk_attention_packed_bf16_best.
 */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_nvfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e2m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e3m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e5m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);

/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_best(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_f16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_best(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_i8_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_nvfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream);

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
 *  The rotation is linear, so FP8 codes rotate at their own scale and need none passed.
 *
 *  @param[in] x Input token matrix of shape @p rows by @p head_count × @p depth.
 *  @param[in] cos Per-token cosine angle grid of shape @p rows by @p depth / 2, shared by heads.
 *  @param[in] sin Per-token sine angle grid of shape @p rows by @p depth / 2, shared across heads.
 *  @param[out] y Output matrix, same shape and dtype as @p x; may alias @p x for in-place rotation.
 *  @param[in] rows The number of token rows.
 *  @param[in] head_count The number of heads per token.
 *  @param[in] depth The even number of channels per head.
 *  @param[in] x_stride Row stride of @p x in bytes.
 *  @param[in] y_stride Row stride of @p y in bytes.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_attention_rope_f32_best(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_capability_t capabilities,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_best(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_serial(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_serial(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_serial(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_serial(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_serial(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_serial(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_serial(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_bf16_t const *keys,
                                                      nk_size_t key_stride, nk_bf16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_f16_t const *keys,
                                                     nk_size_t key_stride, nk_f16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_e4m3_t const *keys,
                                                      nk_size_t key_stride, nk_e4m3_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                    nk_i8_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_nvfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                       nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp4_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                       nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_serial(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_nvfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e2m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e3m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e5m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_nvfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp4_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_serial(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_serial(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                     nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_serial(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                      nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                      nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_serial(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                      nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                      nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                      nk_stream_t stream);

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_haswell(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_haswell(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_bf16_t const *keys,
                                                       nk_size_t key_stride, nk_bf16_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_e4m3_t const *keys,
                                                       nk_size_t key_stride, nk_e4m3_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_haswell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_haswell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_haswell(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                     nk_i8_t const *values, nk_size_t value_stride,
                                                     void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_haswell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_haswell(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                      nk_f32_t *y, nk_size_t rows, nk_size_t head_count,
                                                      nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_haswell(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_haswell(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_skylake(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_skylake(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_bf16_t const *keys,
                                                       nk_size_t key_stride, nk_bf16_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_e4m3_t const *keys,
                                                       nk_size_t key_stride, nk_e4m3_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_skylake(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_skylake(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_skylake(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                      nk_f32_t *y, nk_size_t rows, nk_size_t head_count,
                                                      nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                      nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_skylake(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_skylake(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_icelake(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                     nk_i8_t const *values, nk_size_t value_stride,
                                                     void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_icelake(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_genoa(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_genoa(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_genoa(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_genoa(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_SAPPHIREAMX
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_sapphireamx(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_sapphireamx(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride, nk_bf16_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_sapphireamx(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_sapphireamx(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_sapphireamx(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_sapphireamx(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_sapphireamx(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_sapphireamx(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride, nk_i8_t const *values, nk_size_t value_stride,
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_sapphireamx(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_SAPPHIREAMX

/*  Diamond Rapids AMX provides only the E4M3 attention variant: its native FP8 tiles, driven by
 *  @c _tile_dphf8ps, are its differentiator, while its I8/BF16 paths would merely clone the
 *  Sapphire AMX backend. */
#if NUMKONG_TARGET_DIAMONDAMX
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_diamondamx(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_diamondamx(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_diamondamx(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_DIAMONDAMX

#if NUMKONG_TARGET_SME
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_sme(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride,
                                                   nk_bf16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_bf16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_sme(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride,
                                                   nk_e4m3_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_e4m3_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_sme(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                  nk_f16_t const *values, nk_size_t value_stride,
                                                  void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_f16_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_sme(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_nvfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                    nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_nvfp4_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_sme(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                    nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp4_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys,
                                                        nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys,
                                                        nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys,
                                                        nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys,
                                                        nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_size_t token_count, nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_sme(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                         nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                 nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                 nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                 nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                   nk_size_t depth, nk_u32_t const *query_offsets,
                                                   nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                   nk_size_t keys_after, nk_i8_t const *queries, nk_size_t query_stride,
                                                   void const *key_value_packed, nk_f32_t *output,
                                                   nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_SME

#if NUMKONG_TARGET_NEON
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_neon(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_bf16_t const *keys,
                                                    nk_size_t key_stride, nk_bf16_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_neon(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                   nk_f16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_neon(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_f16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_neon(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_e4m3_t const *keys,
                                                    nk_size_t key_stride, nk_e4m3_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_neon(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                  nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_neon(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_i8_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_neon(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_nvfp4_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_nvfp4_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_neon(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp4_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp4_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_neon(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_neon(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_neon(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_neon(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_neon(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_neon(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_neon(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_neon(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_neon(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_neon(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_neon(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_neon(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_neonbfdot(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride, nk_bf16_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp4_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_neonbfdot(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp4_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp4_cref_t const *keys, nk_size_t key_stride, nk_mxfp4_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp4_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_neonbfdot(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_neonbfdot(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_neonbfdot(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_neonbfdot(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_neonbfdot(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_neonbfdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_neonfhm(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_e4m3_t const *keys,
                                                       nk_size_t key_stride, nk_e4m3_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_neonfhm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_f16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_neonfhm(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_f16_t const *keys,
                                                      nk_size_t key_stride, nk_f16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_neonfhm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_neonfhm(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_nvfp4_best */
NUMKONG_API nk_status_t nk_attention_pack_nvfp4_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                        nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_nvfp4_neonfhm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONFHM

#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_neonsdot(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_i8_t const *keys,
                                                      nk_size_t key_stride, nk_i8_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_neonsdot(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_NEONSDOT

#if NUMKONG_TARGET_RVV
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_rvv(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_rvv(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_size_t token_count, nk_size_t segment_count, nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_rvv(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                         nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride,
                                                   nk_bf16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride,
                                                   nk_e4m3_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_rvv(nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                 nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                 nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                 nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_rvv(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_bf16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_rvv(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_e4m3_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_rvv(nk_size_t head_count, nk_size_t key_value_head_count,
                                                   nk_size_t depth, nk_u32_t const *query_offsets,
                                                   nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                   nk_size_t keys_after, nk_i8_t const *queries, nk_size_t query_stride,
                                                   void const *key_value_packed, nk_f32_t *output,
                                                   nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_v128relaxed(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_v128relaxed(void const *key_value_packed,
                                                                   nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                   nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_v128relaxed(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_v128relaxed(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride, nk_bf16_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_v128relaxed(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_v128relaxed(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride, nk_i8_t const *values, nk_size_t value_stride,
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_v128relaxed(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_v128relaxed(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_v128relaxed(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_V128RELAXED

/*  GPU kernels take their CPU counterparts' arguments and return without waiting on the device.
 *  No entry reads an array on the host, so the key offsets and lengths must be device or managed
 *  memory, like every other pointer.
 *
 *  CUDA on every NVIDIA device: 32 lanes per query row, two sweeps over its keys as the serial
 *  backend makes, and the pack routine every GPU capability shares. */
#if NUMKONG_TARGET_CUDA
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_cuda(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                   nk_f16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_cuda(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_f16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cuda(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_bf16_t const *keys,
                                                    nk_size_t key_stride, nk_bf16_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_cuda(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_cuda(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cuda(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_e4m3_t const *keys,
                                                    nk_size_t key_stride, nk_e4m3_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_cuda(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cuda(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cuda(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                  nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_cuda(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_i8_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_cuda(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_cuda(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_cuda(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
#endif // NUMKONG_TARGET_CUDA

/*  NVIDIA backends from Ampere on: FlashAttention-2 on warp-level `mma.sync` up to depth 256, the
 *  CUDA baseline past it. */
#if NUMKONG_TARGET_AMPERE
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_ampere(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_f16_t const *keys,
                                                     nk_size_t key_stride, nk_f16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_ampere(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_ampere(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_bf16_t const *keys,
                                                      nk_size_t key_stride, nk_bf16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_ampere(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_ampere(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_ampere(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_e4m3_t const *keys,
                                                      nk_size_t key_stride, nk_e4m3_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_ampere(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_ampere(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_ampere(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                    nk_i8_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_ampere(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_AMPERE

/*  NVIDIA Hopper backends, compute capability 9.0, through warpgroup @c wgmma over shared-memory
 *  descriptors. */
#if NUMKONG_TARGET_HOPPER
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_hopper(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_bf16_t const *keys,
                                                      nk_size_t key_stride, nk_bf16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_hopper(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_hopper(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_hopper(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_e4m3_t const *keys,
                                                      nk_size_t key_stride, nk_e4m3_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_hopper(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_hopper(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_hopper(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                    nk_i8_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_hopper(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_HOPPER

/*  NVIDIA datacenter Blackwell backends, the compute capability 10.x family, through single-thread
 *  @c tcgen05 MMAs into tensor memory. */
#if NUMKONG_TARGET_BLACKWELL
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_blackwell(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                        nk_size_t segment_count, nk_f16_t const *keys,
                                                        nk_size_t key_stride, nk_f16_t const *values,
                                                        nk_size_t value_stride, void *key_value_packed,
                                                        nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_blackwell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_blackwell(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_blackwell(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride, nk_bf16_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_blackwell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_blackwell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_blackwell(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_i8_t const *keys,
                                                       nk_size_t key_stride, nk_i8_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_blackwell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_blackwell(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_blackwell(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_blackwell(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_blackwell(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELL

/*  NVIDIA backends for the compute capability 12.x family, with E4M3 on the tensor cores natively.
 *  BF16 and I8 there use the Ampere kernels, which already run at the native rate. */
#if NUMKONG_TARGET_BLACKWELLRTX
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_blackwellrtx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_size_t token_count, nk_size_t segment_count,
                                                                 nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_blackwellrtx(void const *key_value_packed,
                                                                    nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                    nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_blackwellrtx(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_blackwellrtx(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELLRTX

/*  NVIDIA Blackwell Ultra backends, the compute capability 10.3 parts: the Blackwell kernels with
 *  each row's maximum found by @c tcgen05.ld.red as it loads the scores. */
#if NUMKONG_TARGET_BLACKWELLULTRA
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_blackwellultra(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_blackwellultra(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_blackwellultra(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride, nk_bf16_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_blackwellultra(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_blackwellultra(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_blackwellultra(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_size_t token_count, nk_size_t segment_count,
                                                                  nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_blackwellultra(void const *key_value_packed,
                                                                     nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                     nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_blackwellultra(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride, nk_f16_t const *values, nk_size_t value_stride,
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_blackwellultra(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_blackwellultra(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_size_t token_count, nk_size_t segment_count,
                                                                   nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_blackwellultra(void const *key_value_packed,
                                                                      nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                      nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_blackwellultra(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride, nk_e4m3_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_blackwellultra(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_blackwellultra(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_size_t token_count, nk_size_t segment_count,
                                                                 nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_blackwellultra(void const *key_value_packed,
                                                                    nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                    nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_blackwellultra(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride, nk_i8_t const *values, nk_size_t value_stride,
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_blackwellultra(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_BLACKWELLULTRA

/*  ROCm on every AMD device: the CUDA baseline's source, compiled by HIP. */
#if NUMKONG_TARGET_ROCM
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_rocm(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_bf16_t const *keys,
                                                    nk_size_t key_stride, nk_bf16_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_rocm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_rocm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_rocm(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_e4m3_t const *keys,
                                                    nk_size_t key_stride, nk_e4m3_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_rocm(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_rocm(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_rocm(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                  nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_rocm(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_i8_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_ROCM

/*  AMD Instinct MI300 backends, gfx942, through its matrix cores. */
#if NUMKONG_TARGET_CDNA3
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cdna3(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cdna3(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cdna3(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_cdna3(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_cdna3(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cdna3(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cdna3(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cdna3(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                   nk_i8_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_cdna3(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_i8_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA3

/*  AMD Instinct MI350 backends, gfx950, through its matrix cores. */
#if NUMKONG_TARGET_CDNA4
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cdna4(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_cdna4(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_cdna4(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cdna4(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_cdna4(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cdna4(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cdna4(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                   nk_i8_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_cdna4(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_i8_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA4

/*  AMD Instinct MI400 backends, through its matrix cores. */
#if NUMKONG_TARGET_CDNA5
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_cdna5(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_cdna5(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_gradients_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_cdna5(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_cdna5(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_cdna5(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_cdna5(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_cdna5(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                   nk_i8_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_cdna5(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_i8_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_CDNA5

/*  Metal baseline on every Apple GPU: a simdgroup per query row, four sharing each row of a decode
 *  segment, the serial two sweeps past depth 256, and the pack routine every Apple GPU shares. */
#if NUMKONG_TARGET_METAL
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_metal(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_metal(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                    nk_f16_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_metal(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_metal(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                   nk_i8_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_metal(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_i8_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_f32_metal(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_bf16_metal(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                     nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                     nk_stream_t stream);
/** @copydoc nk_attention_rope_f32_best */
NUMKONG_API nk_status_t nk_attention_rope_e4m3_metal(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                     nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                     nk_stream_t stream);
#endif // NUMKONG_TARGET_METAL

/*  Apple family 9 GPUs: Q · K and P · V of 32-row tiles on SIMD-group matrices, operands staged in
 *  @c half or @c bfloat, and the Metal baseline's decode path. */
#if NUMKONG_TARGET_APPLE9
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_apple9(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_bf16_t const *keys,
                                                      nk_size_t key_stride, nk_bf16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_apple9(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_apple9(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_f16_t const *keys,
                                                     nk_size_t key_stride, nk_f16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_apple9(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_apple9(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_apple9(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_e4m3_t const *keys,
                                                      nk_size_t key_stride, nk_e4m3_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_apple9(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_APPLE9

/*  Apple family 10 GPUs: Q · K and P · V of 32-row tiles on the tensor operations of the Neural
 *  Accelerators, I8 summed exactly in integers, and the Metal baseline's decode path. */
#if NUMKONG_TARGET_APPLE10
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_bf16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_apple10(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_bf16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_bf16_t const *keys,
                                                       nk_size_t key_stride, nk_bf16_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_bf16_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_size_f16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_f16_apple10(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_bf16_best */
NUMKONG_API nk_status_t nk_attention_pack_f16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_f16_t const *keys,
                                                      nk_size_t key_stride, nk_f16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_f16_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_e4m3_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_apple10(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_e4m3_best */
NUMKONG_API nk_status_t nk_attention_pack_e4m3_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_e4m3_t const *keys,
                                                       nk_size_t key_stride, nk_e4m3_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_e4m3_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
/** @copydoc nk_attention_pack_size_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_size_i8_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes);
/** @copydoc nk_attention_packed_shape_i8_best */
NUMKONG_API nk_status_t nk_attention_packed_shape_i8_apple10(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream);
/** @copydoc nk_attention_pack_i8_best */
NUMKONG_API nk_status_t nk_attention_pack_i8_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                     nk_i8_t const *values, nk_size_t value_stride,
                                                     void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_stream_t stream);
/** @copydoc nk_attention_packed_bf16_best */
NUMKONG_API nk_status_t nk_attention_packed_i8_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);
#endif // NUMKONG_TARGET_APPLE10

/** Returns the output dtype for attention: accumulator-precision F32 for all inputs. */
NUMKONG_INLINE nk_dtype_t nk_attention_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_bf16_k: return nk_f32_k;
    case nk_f16_k: return nk_f32_k;
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
#include "numkong/attention/neon.h"
#include "numkong/attention/neonbfdot.h"
#include "numkong/attention/neonfhm.h"
#include "numkong/attention/neonsdot.h"
#include "numkong/attention/rvv.h"
#include "numkong/attention/v128.h"
#include "numkong/attention/v128relaxed.h"
#include "numkong/attention/cuda.cuh"
#include "numkong/attention/rocm.cuh"
#include "numkong/attention/ampere.cuh"
#include "numkong/attention/hopper.cuh"
#include "numkong/attention/blackwell.cuh"
#include "numkong/attention/blackwellrtx.cuh"
#include "numkong/attention/blackwellultra.cuh"
#include "numkong/attention/cdna3.cuh"
#include "numkong/attention/cdna4.cuh"
#include "numkong/attention/cdna5.cuh"
#include "numkong/attention/metal.h"
#include "numkong/attention/apple9.h"
#include "numkong/attention/apple10.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_best(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_capability_t capabilities,
                                                            nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_bf16_t const *keys,
                                                    nk_size_t key_stride, nk_bf16_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_best(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_capability_t capabilities,
                                                           nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                   nk_f16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_f16_best(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_f16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_best(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_capability_t capabilities,
                                                            nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_e4m3_t const *keys,
                                                    nk_size_t key_stride, nk_e4m3_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_best(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments,
                                                          nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                  nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_best(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_i8_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_best(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_capability_t capabilities,
                                                             nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_nvfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_nvfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_nvfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_best(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_capability_t capabilities,
                                                             nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp4_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                     nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp4_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e2m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp6e3m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e4m3_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_capability_t capabilities, nk_size_t *bytes) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(token_count), nk_unused_(segment_count),
        nk_unused_(capabilities), nk_unused_(bytes);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_best(void const *key_value_packed,
                                                                 nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                 nk_size_t *segments, nk_capability_t capabilities,
                                                                 nk_stream_t stream) {
    nk_unused_(key_value_packed), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(segments),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_best(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                         nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys,
                                                         nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
                                                         nk_size_t value_stride, void *key_value_packed,
                                                         nk_size_t tasks_begin, nk_size_t tasks_end,
                                                         nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(key_offsets), nk_unused_(key_lengths),
        nk_unused_(segment_count), nk_unused_(keys), nk_unused_(key_stride), nk_unused_(values),
        nk_unused_(value_stride), nk_unused_(key_value_packed), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(tasks_begin), nk_unused_(tasks_end),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_mxfp8e5m2_best(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output,
    nk_f32_t const *output_gradient, nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient,
    nk_size_t query_gradient_stride, nk_f32_t *key_gradient, nk_f32_t *value_gradient,
    nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end, nk_capability_t capabilities,
    nk_stream_t stream) {
    nk_unused_(head_count), nk_unused_(key_value_head_count), nk_unused_(depth), nk_unused_(query_offsets),
        nk_unused_(query_token_count), nk_unused_(scale), nk_unused_(keys_before), nk_unused_(keys_after),
        nk_unused_(queries), nk_unused_(query_stride), nk_unused_(key_value_packed), nk_unused_(output),
        nk_unused_(output_gradient), nk_unused_(output_stride), nk_unused_(log_sum_exp), nk_unused_(query_gradient),
        nk_unused_(query_gradient_stride), nk_unused_(key_gradient), nk_unused_(value_gradient),
        nk_unused_(key_value_gradient_stride), nk_unused_(tasks_begin), nk_unused_(tasks_end), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_best(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_capability_t capabilities,
                                                   nk_stream_t stream) {
    nk_unused_(x), nk_unused_(cos), nk_unused_(sin), nk_unused_(y), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride), nk_unused_(y_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_best(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(x), nk_unused_(cos), nk_unused_(sin), nk_unused_(y), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride), nk_unused_(y_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_best(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride,
                                                    nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(x), nk_unused_(cos), nk_unused_(sin), nk_unused_(y), nk_unused_(rows), nk_unused_(head_count),
        nk_unused_(depth), nk_unused_(x_stride), nk_unused_(y_stride), nk_unused_(capabilities), nk_unused_(stream);
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
