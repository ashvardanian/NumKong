/**
 *  @brief Panel-based FlashAttention kernels for Intel Sapphire Rapids AMX.
 *  @file include/numkong/attention/sapphireamx.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Implements bidirectional (non-causal) scaled dot-product attention over a @b ragged
 *  batch of independent segments:
 *
 *      O[s] = softmax(Q[s] × K[s]ᵀ · scale) × V[s]      for each segment s
 *
 *  BF16, E4M3, and I8 inputs, F32 outputs, any `depth` from 1 to 256: channels are
 *  zero-padded to 32-deep AMX tiles internally (exact math — padded K channels add zero
 *  to every score, padded V channels are never stored back), and `depth > 256` routes
 *  to the width-agnostic serial tier from all entry points, so packing and attention
 *  always agree on the buffer format. E4M3 has no AMX tile type on Sapphire Rapids, so
 *  it widens to BF16 at the boundaries — K/V during packing, Q during tile loads — and
 *  shares every instruction of the BF16 panel pipeline (the `dots` family's E4M3 recipe).
 *  I8 keeps its own pipeline shape-for-shape: exact I32 scores via TDPBSSD over
 *  64-channel-deep quad-interleaved tiles, softmax weights quantized to U8 as
 *  `round(255 · 2^(s₂ − m₂))` (the running-max position lands on 255, so the weight sum
 *  is never zero and the 255 cancels in normalization), and P×V via TDPBUSD — the serial
 *  I8 tier defines that contract.
 *
 *  The ragged form is the only form: transformer inference packs many variable-length
 *  segments into one token buffer (UForm's `segment_offsets` / `segment_lengths`
 *  convention), and a single-segment call is just `segment_count == 1`. Grouped-query
 *  attention is supported via `num_kv_heads`; cross-attention and pooling fall out of
 *  per-segment query counts that differ from the packed KV lengths — the attention pool
 *  is `query_offsets = {0, 1, 2, …}`, one learned query per segment. Causal masking is
 *  deliberately a separate (future) symbol.
 *
 *  @section attention_sapphireamx_layout Tensor Layout
 *
 *  Q, K, V, and O all use the activations-natural layout `[tokens, heads × depth]` with
 *  an explicit row stride in bytes: element `(head, token, channel)` lives at
 *  `base + token · stride + (head · depth + channel) · sizeof(scalar)`.
 *  A fused QKV projection output `[tokens, 3 × hidden]` is therefore consumable in place —
 *  pass interior pointers and `3 × hidden × sizeof(scalar)` strides, no copies.
 *
 *  Both packing and attention take a task window over the flat `segment × head` grid
 *  (`first_task`, `task_count`, with `task_count == 0` meaning "through the end"), so a
 *  parallel caller distributes tasks across threads — one thread per physical core,
 *  longest segments first — without any second entry point. Tasks touch disjoint outputs;
 *  packing tasks write disjoint tile blocks (the tiny header/directory bytes are written
 *  identically by every packing call).
 *
 *  @section attention_sapphireamx_design Design
 *
 *  Classic FlashAttention tiling (Bᶜ = 32) interleaves a 16×32 score block, an online
 *  softmax, and a P×V accumulation. Ablation on that sequence shows the AMX work plus all
 *  tile moves cost ~275 cycles per block while the online softmax chain costs ~600 (the
 *  horizontal `reduce_max`/`reduce_add` chains, not the exponent) and the TMM→memory→ZMM
 *  output round-trip another ~400 — the matrix unit idles ~85% of the time. This kernel
 *  instead sweeps KV in @b panels of 512 columns — FlashAttention with Bᶜ = 512,
 *  mathematically exact — with three structural choices, each validated by A/B measurement
 *  against the alternative it replaced:
 *
 *  1. @b 32-row register blocking. Scores accumulate as a 2×2 grid of TMM tiles
 *     (two 16-row Q tiles × two 16-column K tiles), so every loaded operand tile feeds
 *     two MACs: one tile load per `tdpbf16ps` in both matmuls, versus 1.5 for the 16-row
 *     variant this replaced, and all per-query-block fixed costs amortize over twice the
 *     rows. P×V holds four F32 accumulator tiles TMM-resident per 32-channel slice of
 *     the head across the panel's whole depth; the accumulator crosses into ZMM once per
 *     panel per slice, fused with the online correction as a single FMA:
 *     `O = O · 2^(m_old − m_new) + O_panel`.
 *  2. @b KV-reuse chunking. Four 32-row query blocks share each KV panel sweep, cutting
 *     packed-K/V traffic 4× — at 16K-token sequences (the Large-tier segment budget),
 *     where K+V no longer fit in L2, this alone is worth ~1.45×.
 *  3. @b Base-2 streaming softmax. `softmax(x) = softmax₂(x · log₂e)`, with log₂e folded
 *     into the score scale, so the exponent is `2^r` with an @b exact range reduction
 *     (`r = x − round(x)`; no Cody-Waite ladder) and a shorter polynomial. The pass is
 *     row-major with running sums in registers: one horizontal reduction per row per
 *     panel, no per-block online state. The exponent's lower clamp is −125, keeping the
 *     smallest result `2⁻¹²⁵` a @b normal float: values in the last 1.7 octaves above
 *     F32's true limit are denormal, and one denormal operand in the correction FMA costs
 *     a ~150-cycle FP assist per element — measured as a 2.5× whole-kernel slowdown.
 *
 *  Rejected by measurement: exp/P×V instruction interleaving (wins at 16-row blocking,
 *  loses at 32-row where the vector stage dominates each step), sigmoid scoring (the
 *  division costs what the max/sum bookkeeping saves), software prefetching.
 *
 *  Measured on a single Sapphire Rapids core (Xeon 8468, ~3.2 GHz sustained), BF16,
 *  `depth = 128`: ~0.63 TFLOPS at `q = kv = 256`, ~0.82 at 1024², ~0.80 at 4096²,
 *  ~0.81 at 16384², against ~3.4 TFLOPS `tdpbf16ps` peak and ~0.36 TFLOPS for the classic
 *  Bᶜ=32 flash schedule this file previously drafted. On a ragged 52-segment batch
 *  (128–4096-token mix), task-parallel scaling reaches ~96% efficiency at one thread per
 *  physical core.
 *
 *  @section attention_sapphireamx_packing Packed KV Format
 *
 *  `[64 B header][directory][segment 0 tiles][segment 1 tiles]…`, where the directory is
 *  `nk_u64_t tile_offsets[segment_count + 1]` (bytes, relative to the first tile block;
 *  the last entry is the total) followed by `nk_u32_t segment_lengths[segment_count]`,
 *  padded to 64 bytes. Every segment's sequence is zero-padded to a multiple of 32 and
 *  its channels to a multiple of 32; each block holds K tiles for all KV heads, then V
 *  tiles, always as BF16 regardless of the input dtype. K is packed transposed for Q×Kᵀ
 *  as pair-interleaved B-tiles `[kv_tile][depth_tile]` (`kv_tile` = 16 positions,
 *  `depth_tile` = 32 channels). V is packed for P×V as pair-interleaved B-tiles
 *  `[depth_tile_idx][position_block_idx]` — depth-major so each output tile's depth accumulation streams
 *  contiguous 1 KB tiles (one depth tile = 16 channels, one position block = 32 positions).
 *  Zero-padded K rows yield zero scores, which the masked softmax turns into zero
 *  weights; padded V rows and channels are multiplied by those zero weights or skipped
 *  at the final store.
 *
 *  The kernel keeps ~350 KB of scratch on the stack: a 64 KB F32 score panel, a 32 KB
 *  BF16 weight panel, four 32 KB output accumulators, and 64 KB of packed Q tiles.
 *
 *  @section attention_sapphireamx_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Sapphire
 *      _tile_dpbf16ps              TDPBF16PS (TMM, TMM, TMM)       16cy throughput (16×16×32)
 *      _tile_loadd                 TILELOADD (TMM, MEM)            ~8cy @ p23, ~45cy latency
 *      _tile_stored                TILESTORED (MEM, TMM)           ~16cy @ p49
 *      _mm512_cvtneps_pbh          VCVTNEPS2BF16 (YMM, ZMM)        4cy @ p05
 *      _mm256_permutex2var_epi16   VPERMT2W (YMM, YMM, YMM)        3cy @ p5
 */
#ifndef NK_ATTENTION_SAPPHIREAMX_H
#define NK_ATTENTION_SAPPHIREAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_SAPPHIREAMX

#include "numkong/attention/serial.h"  // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_exp2_f32x16_skylake_` — AMX implies the Skylake tier
#include "numkong/reduce/skylake.h"    // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`
#include "numkong/dots/sapphireamx.h"  // `nk_amx_tile_configure_sapphireamx_`, tile transposer, FP8 loaders

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                       \
    __attribute__((target(                                                                                                          \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,avx512bf16,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "avx512bf16", \
                   "f16c", "fma", "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8")
#endif

enum {
    /** KV panel width in positions; the 32-row F32 score panel (64 KB) stays L2-resident. */
    nk_attention_panel_sapphireamx_k_ = 512,
    /** 32-row query blocks sharing one KV panel sweep; bounds packed-K/V re-reads at long context. */
    nk_attention_chunk_sapphireamx_k_ = 4,
    /** Widest head this backend handles in tiles; larger heads route to the serial tier. */
    nk_attention_max_depth_sapphireamx_k_ = 256,
};

/** @brief Gathers one 16×32 A-tile of BF16 from strided rows, zero-padding rows and channels. */
typedef void (*nk_attention_gather_sapphireamx_t_)(nk_dots_bf16_a16x32_sapphireamx_t *tile, void const *source,
                                                   nk_size_t source_stride_bytes, nk_size_t valid_positions,
                                                   nk_size_t valid_columns);

NK_HELPER_INLINE void nk_attention_gather_bf16_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *tile, void const *source,
                                                            nk_size_t source_stride_bytes, nk_size_t valid_positions,
                                                            nk_size_t valid_columns) {
    nk_dots_bf16_load_a_sapphireamx_(tile, (nk_bf16_t const *)source, source_stride_bytes / sizeof(nk_bf16_t),
                                     valid_positions, valid_columns);
}

NK_HELPER_INLINE void nk_attention_gather_e4m3_sapphireamx_(nk_dots_bf16_a16x32_sapphireamx_t *tile, void const *source,
                                                            nk_size_t source_stride_bytes, nk_size_t valid_positions,
                                                            nk_size_t valid_columns) {
    nk_dots_e4m3_load_a_sapphireamx_(tile, (nk_e4m3_t const *)source, source_stride_bytes / sizeof(nk_e4m3_t),
                                     valid_positions, valid_columns);
}

/** @brief Loads two V rows of 16 channels as BF16 for pair-interleaving; dead rows/channels zero. */
typedef void (*nk_attention_v_rows_sapphireamx_t_)(void const *row_a, void const *row_b, int a_live, int b_live,
                                                   __mmask16 columns_m16, __m256i *a_bf16x16, __m256i *b_bf16x16);

NK_HELPER_INLINE void nk_attention_v_rows_bf16_sapphireamx_(void const *row_a, void const *row_b, int a_live,
                                                            int b_live, __mmask16 columns_m16, __m256i *a_bf16x16,
                                                            __m256i *b_bf16x16) {
    *a_bf16x16 = a_live ? _mm256_maskz_loadu_epi16(columns_m16, row_a) : _mm256_setzero_si256();
    *b_bf16x16 = b_live ? _mm256_maskz_loadu_epi16(columns_m16, row_b) : _mm256_setzero_si256();
}

/** @brief One 32-lane E4M3→BF16 conversion covers both rows, packed into the two 128-bit halves. */
NK_HELPER_INLINE void nk_attention_v_rows_e4m3_sapphireamx_(void const *row_a, void const *row_b, int a_live,
                                                            int b_live, __mmask16 columns_m16, __m256i *a_bf16x16,
                                                            __m256i *b_bf16x16) {
    __m128i a_e4m3x16 = a_live ? _mm_maskz_loadu_epi8(columns_m16, row_a) : _mm_setzero_si128();
    __m128i b_e4m3x16 = b_live ? _mm_maskz_loadu_epi8(columns_m16, row_b) : _mm_setzero_si128();
    __m256i both_e4m3x32 = _mm256_inserti128_si256(_mm256_castsi128_si256(a_e4m3x16), b_e4m3x16, 1);
    __m512i both_bf16x32 = nk_e4m3x32_to_bf16x32_icelake_(both_e4m3x32);
    *a_bf16x16 = _mm512_castsi512_si256(both_bf16x32);
    *b_bf16x16 = _mm512_extracti64x4_epi64(both_bf16x32, 1);
}

NK_HELPER_INLINE nk_size_t nk_attention_packed_size_sapphireamx_(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_u32_t const *segment_lengths,
                                                                 nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t total_tile_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 32);
        total_tile_bytes += 2 * key_value_head_count * position_count_padded * depth_padded *
                            sizeof(nk_bf16_t); // K + V
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + total_tile_bytes;
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_bf16_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                    nk_u32_t const *segment_lengths,
                                                                    nk_size_t segment_count) {
    // Shapes outside the AMX fast-path envelope route to the width-agnostic serial tier;
    // the rule is a pure function of the arguments, so pack and attention always agree.
    if (depth > nk_attention_max_depth_sapphireamx_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_sapphireamx_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_e4m3_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                    nk_u32_t const *segment_lengths,
                                                                    nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_sapphireamx_(key_value_head_count, depth, segment_lengths, segment_count);
}

/**
 *  @brief Shared packing core: K/V of any dtype become BF16 AMX tiles via the two loaders.
 *  The header and directory are deterministic functions of the arguments, so concurrent
 *  packing tasks may rewrite them with identical bytes.
 */
NK_HELPER_INLINE void nk_attention_pack_sapphireamx_(                                     //
    void const *keys, void const *values, nk_size_t element_bytes,                        //
    nk_attention_gather_sapphireamx_t_ gather, nk_attention_v_rows_sapphireamx_t_ v_rows, //
    nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                     //
    nk_size_t segment_count,                                                              //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed,     //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const tile_elements = 512;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t const depth_tiles = depth_padded / 32;
    nk_size_t const dim_tiles = depth_padded / 16;

    // Header + directory: deterministic from the arguments, safe to rewrite from any task.
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 32, depth_padded * sizeof(nk_bf16_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *tile_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *tiles_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    __m256i const interleave_index_u16x16 = _mm256_setr_epi16( //
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m256i const interleave_index_high_u16x16 = _mm256_setr_epi16( //
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 32);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded * sizeof(nk_bf16_t);

        // K: gather 16 (strided) rows × 32 channels as BF16, transpose into the pair-interleaved
        // B-tile the same way the `dots` packer does, tiles ordered [kv_tile][depth_tile].
        nk_bf16_t *keys_head_tiles = (nk_bf16_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                                   key_value_head_idx * bytes_per_head);
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / 16; position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * 16;
            nk_size_t const valid_positions = (position_start + 16 <= position_count) ? 16
                                              : (position_start < position_count)     ? position_count - position_start
                                                                                      : 0;
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
                nk_size_t const channel_start = depth_tile * 32;
                nk_size_t const valid_columns = (channel_start + 32 <= depth) ? 32
                                                : (channel_start < depth)     ? depth - channel_start
                                                                              : 0;
                char const *source = (char const *)keys + (position_first + position_start) * key_stride_bytes +
                                     (key_value_head_idx * depth + channel_start) * element_bytes;
                nk_dots_bf16_a16x32_sapphireamx_t source_tile;
                nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
                gather(&source_tile, source, key_stride_bytes, valid_positions, valid_columns);
                nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
                nk_bf16_t *tile_output = keys_head_tiles +
                                         (position_tile_idx * depth_tiles + depth_tile) * tile_elements;
                for (nk_size_t i = 0; i < tile_elements * sizeof(nk_bf16_t); i += 64)
                    _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
            }
        }

        // V: interleave consecutive position pairs per 16-channel column group with one
        // VPERMT2W per pair, tiles ordered [depth_tile][position_block] (depth-major streaming for P×V).
        nk_bf16_t *values_head_tiles = (nk_bf16_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                                     (key_value_head_count + key_value_head_idx) * bytes_per_head);
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < dim_tiles; depth_tile_idx++) {
            nk_size_t const channel_start = depth_tile_idx * 16;
            nk_size_t const valid_columns = (channel_start + 16 <= depth) ? 16
                                            : (channel_start < depth)     ? depth - channel_start
                                                                          : 0;
            __mmask16 const columns_m16 = (__mmask16)((1u << valid_columns) - 1);
            for (nk_size_t position_block_idx = 0; position_block_idx < position_count_padded / 32;
                 position_block_idx++) {
                nk_bf16_t *tile_output = values_head_tiles +
                                         (depth_tile_idx * (position_count_padded / 32) + position_block_idx) *
                                             tile_elements;
                for (nk_size_t pair = 0; pair < 16; pair++) {
                    nk_size_t const row_a = position_block_idx * 32 + pair * 2, row_b = row_a + 1;
                    char const *row_a_ptr = (char const *)values + (position_first + row_a) * value_stride_bytes +
                                            (key_value_head_idx * depth + channel_start) * element_bytes;
                    char const *row_b_ptr = (char const *)values + (position_first + row_b) * value_stride_bytes +
                                            (key_value_head_idx * depth + channel_start) * element_bytes;
                    __m256i a_bf16x16, b_bf16x16;
                    v_rows(row_a_ptr, row_b_ptr, row_a < position_count, row_b < position_count, columns_m16,
                           &a_bf16x16, &b_bf16x16);
                    __m256i low_bf16x16 = _mm256_permutex2var_epi16(a_bf16x16, interleave_index_u16x16, b_bf16x16);
                    __m256i high_bf16x16 = _mm256_permutex2var_epi16(a_bf16x16, interleave_index_high_u16x16,
                                                                     b_bf16x16);
                    // Unaligned: the packed blob may live in a Python flex-array with no 32 B guarantee.
                    _mm256_storeu_si256((__m256i *)(tile_output + pair * 32), low_bf16x16);
                    _mm256_storeu_si256((__m256i *)(tile_output + pair * 32 + 16), high_bf16x16);
                }
            }
        }
    }
    nk_compiler_barrier_sapphireamx_();
}

NK_API_COMPTIME void nk_attention_pack_bf16_sapphireamx(                              //
    nk_bf16_t const *keys, nk_bf16_t const *values,                                   //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_sapphireamx_(keys, values, sizeof(nk_bf16_t), &nk_attention_gather_bf16_sapphireamx_,
                                   &nk_attention_v_rows_bf16_sapphireamx_, key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                   key_value_packed, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_pack_e4m3_sapphireamx(                              //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                                   //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_sapphireamx_(keys, values, sizeof(nk_e4m3_t), &nk_attention_gather_e4m3_sapphireamx_,
                                   &nk_attention_v_rows_e4m3_sapphireamx_, key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                   key_value_packed, first_task, task_count);
}

/** @brief Per-call scratch: score/weight panels, output accumulators, packed Q tiles. */
typedef struct {
    NK_ALIGN64 nk_f32_t scores_panel[32 * nk_attention_panel_sapphireamx_k_];   // 64 KB
    NK_ALIGN64 nk_bf16_t weights_panel[32 * nk_attention_panel_sapphireamx_k_]; // 32 KB
    NK_ALIGN64 nk_f32_t o_acc[nk_attention_chunk_sapphireamx_k_][32 * nk_attention_max_depth_sapphireamx_k_];
    nk_dots_bf16_a16x32_sapphireamx_t q_tiles[nk_attention_chunk_sapphireamx_k_][2][8]; // 64 KB
} nk_attention_scratch_sapphireamx_t;

/**
 *  @brief Stage 1: Q×Kᵀ for one KV panel with 2×2 register blocking, then the row-max sweep.
 *
 *  Per 32-column pair, scores accumulate in TMM0-3 (two 16-row Q tiles × two 16-column
 *  K tiles) with Q in TMM4/5 and K in TMM6/7 — every loaded tile feeds two MACs. The
 *  depth loop covers `depth_tiles` 32-channel steps (4 for the `depth = 128` hot path).
 *  Emits per-row raw (unscaled) maxima for both 16-row groups; padded columns hold exact
 *  zeros, which can only raise the max — always numerically safe, cancels in normalization.
 */
NK_HELPER_INLINE void nk_attention_score_panel_sapphireamx_(      //
    nk_dots_bf16_a16x32_sapphireamx_t const (*q_tiles)[8],        // [2][depth_tiles] pre-packed Q
    nk_bf16_t const *keys_head_tiles, nk_size_t panel_first_tile, // K tiles, [kv_tile][depth_tile]
    nk_size_t panel_pairs, nk_size_t depth_tiles,                 // panel width / 32, channels / 32
    nk_f32_t *scores_panel, nk_size_t panel_width,                // [32][panel_width]
    nk_f32_t (*panel_max)[16]) {                                  // [2][16] raw row maxima out

    int const panel_stride_bytes = (int)(panel_width * sizeof(nk_f32_t));
    nk_size_t const k_tile_stride = depth_tiles * 512;
    for (nk_size_t pair = 0; pair < panel_pairs; pair++) {
        nk_bf16_t const *k_tile0 = keys_head_tiles + (panel_first_tile + pair * 2) * k_tile_stride;
        nk_bf16_t const *k_tile1 = k_tile0 + k_tile_stride;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
            _tile_loadd(4, q_tiles[0][depth_tile].data, 64);
            _tile_loadd(5, q_tiles[1][depth_tile].data, 64);
            _tile_loadd(6, k_tile0 + depth_tile * 512, 64);
            _tile_loadd(7, k_tile1 + depth_tile * 512, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_stored(0, scores_panel + pair * 32, panel_stride_bytes);
        _tile_stored(1, scores_panel + pair * 32 + 16, panel_stride_bytes);
        _tile_stored(2, scores_panel + 16 * panel_width + pair * 32, panel_stride_bytes);
        _tile_stored(3, scores_panel + 16 * panel_width + pair * 32 + 16, panel_stride_bytes);
    }
    for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
        for (nk_size_t row = 0; row < 16; row++) {
            nk_f32_t const *scores_row = scores_panel + (row_tile_idx * 16 + row) * panel_width;
            __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
            for (nk_size_t channel_idx = 0; channel_idx < panel_pairs * 32; channel_idx += 16)
                max_f32x16 = _mm512_max_ps(max_f32x16, _mm512_loadu_ps(scores_row + channel_idx));
            panel_max[row_tile_idx][row] = nk_reduce_max_f32x16_skylake_(max_f32x16);
        }
}

/**
 *  @brief Stage 2: streaming base-2 softmax over one panel (32 rows).
 *
 *  Row-major single pass: `weights = bf16(2^(scores · scale₂ − m₂_row))` with the running
 *  sum in a register — one horizontal reduction per row per panel, no per-block online
 *  state. Columns past `valid_cols` (sequence padding) get zero weights and no sum
 *  contribution. `new_max` is the base-2-scaled running maximum, already merged.
 */
NK_HELPER_INLINE void nk_attention_exp_panel_sapphireamx_(                 //
    nk_f32_t const *scores_panel, nk_bf16_t *weights_panel,                // [32][panel] each
    nk_size_t panel_cols, nk_size_t valid_channels, nk_size_t panel_width, //
    nk_f32_t scale2, nk_f32_t const (*new_max)[16], nk_f32_t (*panel_sums)[16]) {

    __m512 const scale_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const full_cols = valid_channels & ~(nk_size_t)15;
    __mmask16 const tail_m16 = (__mmask16)((1u << (valid_channels - full_cols)) - 1);
    for (nk_size_t row = 0; row < 32; row++) {
        nk_f32_t const *scores_row = scores_panel + row * panel_width;
        nk_bf16_t *weights_row = weights_panel + row * panel_width;
        __m512 const max_f32x16 = _mm512_set1_ps(new_max[row / 16][row % 16]);
        __m512 sum_f32x16 = _mm512_setzero_ps();
        nk_size_t channel_idx = 0;
        for (; channel_idx < full_cols; channel_idx += 16) {
            __m512 exp_f32x16 = nk_attention_exp2_f32x16_skylake_(
                _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + channel_idx), scale_f32x16, max_f32x16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, exp_f32x16);
            _mm256_store_si256((__m256i *)(weights_row + channel_idx), (__m256i)_mm512_cvtneps_pbh(exp_f32x16));
        }
        if (channel_idx < valid_channels) {
            __m512 exp_f32x16 = _mm512_maskz_mov_ps(
                tail_m16, nk_attention_exp2_f32x16_skylake_(
                              _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + channel_idx), scale_f32x16, max_f32x16)));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, exp_f32x16);
            _mm256_store_si256((__m256i *)(weights_row + channel_idx), (__m256i)_mm512_cvtneps_pbh(exp_f32x16));
            channel_idx += 16;
        }
        for (; channel_idx < panel_cols; channel_idx += 16) // zero weights for the padded remainder
            _mm256_store_si256((__m256i *)(weights_row + channel_idx), _mm256_setzero_si256());
        panel_sums[row / 16][row % 16] = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    }
}

/**
 *  @brief Stage 3: P×V for one panel with 2×2 register blocking, TMM-resident accumulation.
 *
 *  One sweep over the weight panel per 32-channel slice of the head: TMM0-3 hold
 *  (two 16-row P groups) × (two 16-channel V tiles) across the panel's whole depth, with
 *  P in TMM4/5 and V in TMM6/7 — one tile load per MAC. Each accumulator crosses into
 *  the F32 output accumulator once per panel, fused with the online correction:
 *  `o_acc = o_acc · correction + o_panel`.
 */
NK_HELPER_INLINE void nk_attention_weighted_sum_panel_sapphireamx_( //
    nk_bf16_t const *weights_panel, nk_size_t panel_width,          // [32][panel] BF16
    nk_bf16_t const *values_head_tiles, nk_size_t position_blocks,  // V tiles, [depth_tile][position_block]
    nk_size_t panel_first_step, nk_size_t panel_steps,              //
    nk_size_t depth_blocks, nk_size_t output_stride_floats,         // channels / 32, o_acc row stride
    nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {           // [2][16], [32][o_stride]

    NK_ALIGN64 nk_f32_t o_panel_tile[16][16];
    int const weights_stride_bytes = (int)(panel_width * sizeof(nk_bf16_t));
    for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_blocks; depth_tile_idx++) {
        nk_bf16_t const *v_tile0 = values_head_tiles +
                                   ((depth_tile_idx * 2 + 0) * position_blocks + panel_first_step) * 512;
        nk_bf16_t const *v_tile1 = values_head_tiles +
                                   ((depth_tile_idx * 2 + 1) * position_blocks + panel_first_step) * 512;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t step = 0; step < panel_steps; step++) {
            _tile_loadd(4, weights_panel + step * 32, weights_stride_bytes);
            _tile_loadd(5, weights_panel + 16 * panel_width + step * 32, weights_stride_bytes);
            _tile_loadd(6, v_tile0 + step * 512, 64);
            _tile_loadd(7, v_tile1 + step * 512, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        for (nk_size_t tile = 0; tile < 4; tile++) {
            switch (tile) { // `_tile_stored` requires an immediate tile register index
            case 0: _tile_stored(0, o_panel_tile, 64); break;
            case 1: _tile_stored(1, o_panel_tile, 64); break;
            case 2: _tile_stored(2, o_panel_tile, 64); break;
            case 3: _tile_stored(3, o_panel_tile, 64); break;
            }
            nk_size_t const row_tile_idx = tile / 2;
            nk_size_t const channel_start = depth_tile_idx * 32 + (tile % 2) * 16;
            for (nk_size_t row = 0; row < 16; row++) {
                __m512 const correction_f32x16 = _mm512_set1_ps(corrections[row_tile_idx][row]);
                __m512 accumulator_f32x16 = _mm512_load_ps(o_acc + (row_tile_idx * 16 + row) * output_stride_floats +
                                                           channel_start);
                accumulator_f32x16 = _mm512_fmadd_ps(accumulator_f32x16, correction_f32x16,
                                                     _mm512_load_ps(o_panel_tile[row]));
                _mm512_store_ps(o_acc + (row_tile_idx * 16 + row) * output_stride_floats + channel_start,
                                accumulator_f32x16);
            }
        }
    }
}

/**
 *  @brief One (segment, head) task: panel-flash attention for `query_count` rows against
 *         one segment's packed KV, with KV-reuse chunking over 128-query groups.
 */
NK_HELPER_INLINE void nk_attention_task_sapphireamx_(                                           //
    void const *queries, nk_size_t element_bytes, nk_attention_gather_sapphireamx_t_ gather,    //
    nk_f32_t *output,                                                                           //
    nk_bf16_t const *keys_head_tiles, nk_bf16_t const *values_head_tiles,                       //
    nk_size_t head, nk_size_t depth, nk_size_t position_count, nk_size_t position_count_padded, //
    nk_size_t query_first, nk_size_t row_count,                                                 //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale2,               //
    nk_attention_scratch_sapphireamx_t *scratch) {

    nk_size_t const panel_width = nk_attention_panel_sapphireamx_k_;
    nk_size_t const row_blocks = nk_attention_chunk_sapphireamx_k_;
    nk_size_t const kv_padded = nk_size_round_up_to_multiple_(position_count, 32);
    nk_size_t const position_blocks = position_count_padded / 32;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t const depth_tiles = depth_padded / 32;
    nk_size_t const output_stride_floats = depth_padded;
    nk_size_t const o_stride_out = output_stride_bytes / sizeof(nk_f32_t);

    NK_ALIGN64 nk_f32_t panel_max[2][16];
    NK_ALIGN64 nk_f32_t corrections[2][16];
    NK_ALIGN64 nk_f32_t new_max_arr[2][16];
    NK_ALIGN64 nk_f32_t panel_sums[2][16];
    __m512 row_max2_f32x16[nk_attention_chunk_sapphireamx_k_][2]; // base-2-scaled running maxima
    __m512 row_sum_f32x16[nk_attention_chunk_sapphireamx_k_][2];
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const dim_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const blocks = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                     ? (row_count - row_block_start + 31) / 32
                                     : row_blocks;
        for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const position_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                nk_size_t const valid_positions = position_start >= row_count          ? 0
                                                  : (row_count - position_start >= 16) ? 16
                                                                                       : row_count - position_start;
                for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
                    nk_size_t const channel_start = depth_tile * 32;
                    nk_size_t const valid_columns = (channel_start + 32 <= depth) ? 32 : depth - channel_start;
                    gather(&scratch->q_tiles[row_block_idx][row_tile_idx][depth_tile],
                           (char const *)queries + (query_first + position_start) * query_stride_bytes +
                               (head * depth + channel_start) * element_bytes,
                           query_stride_bytes, valid_positions, valid_columns);
                }
                row_max2_f32x16[row_block_idx][row_tile_idx] = _mm512_set1_ps(NK_F32_MIN);
                row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_setzero_ps();
            }

        // Every query block of the chunk consumes each KV panel before the next panel is
        // touched, bounding packed-K/V re-reads to (query_count / 128) sweeps.
        for (nk_size_t panel_start = 0; panel_start < kv_padded; panel_start += panel_width) {
            nk_size_t const panel_cols = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                  : (kv_padded - panel_start);
            nk_size_t const valid_channels = (panel_start + panel_cols <= position_count)
                                                 ? panel_cols
                                                 : (position_count - panel_start);

            for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++) {
                nk_attention_score_panel_sapphireamx_(
                    (nk_dots_bf16_a16x32_sapphireamx_t const(*)[8])scratch->q_tiles[row_block_idx], keys_head_tiles,
                    panel_start / 16, panel_cols / 32, depth_tiles, scratch->scores_panel, panel_width, panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512 const old_max_f32x16 = row_max2_f32x16[row_block_idx][row_tile_idx];
                    __m512 const new_max_f32x16 = _mm512_max_ps(
                        old_max_f32x16, _mm512_mul_ps(_mm512_load_ps(panel_max[row_tile_idx]), scale2_f32x16));
                    __m512 const corrections_f32x16 = nk_attention_exp2_f32x16_skylake_(
                        _mm512_sub_ps(old_max_f32x16, new_max_f32x16));
                    row_max2_f32x16[row_block_idx][row_tile_idx] = new_max_f32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corrections_f32x16);
                    _mm512_store_ps(new_max_arr[row_tile_idx], new_max_f32x16);
                }
                nk_attention_exp_panel_sapphireamx_(scratch->scores_panel, scratch->weights_panel, panel_cols,
                                                    valid_channels, panel_width, scale2,
                                                    (nk_f32_t const(*)[16])new_max_arr, panel_sums);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
                    row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_fmadd_ps(
                        row_sum_f32x16[row_block_idx][row_tile_idx], _mm512_load_ps(corrections[row_tile_idx]),
                        _mm512_load_ps(panel_sums[row_tile_idx]));
                nk_attention_weighted_sum_panel_sapphireamx_(
                    scratch->weights_panel, panel_width, values_head_tiles, position_blocks, panel_start / 32,
                    panel_cols / 32, depth_tiles, output_stride_floats, (nk_f32_t const(*)[16])corrections,
                    scratch->o_acc[row_block_idx]);
            }
        }

        for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const position_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                if (position_start >= row_count) break;
                nk_size_t const valid_positions = (row_count - position_start >= 16) ? 16 : row_count - position_start;
                NK_ALIGN64 nk_f32_t row_sums[16];
                _mm512_store_ps(row_sums, row_sum_f32x16[row_block_idx][row_tile_idx]);
                for (nk_size_t row = 0; row < valid_positions; row++) {
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(1.0f / row_sums[row]);
                    nk_f32_t const *accumulator_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + position_start + row) * o_stride_out + head * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(
                            output_row + channel_idx, dim_tail_m16,
                            _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                }
                for (nk_size_t row = 0; row < 16; row++)
                    for (nk_size_t channel_idx = 0; channel_idx < output_stride_floats; channel_idx += 16)
                        _mm512_store_ps(&scratch->o_acc[row_block_idx]
                                                       [(row_tile_idx * 16 + row) * output_stride_floats + channel_idx],
                                        zero_f32x16);
            }
    }
}

/**
 *  @brief Shared entry: resolves the task window and per-segment tile bases, then runs tasks.
 */
NK_HELPER_INLINE void nk_attention_packed_sapphireamx_(                                                         //
    void const *queries, nk_size_t element_bytes, nk_attention_gather_sapphireamx_t_ gather,                    //
    void const *key_value_packed, nk_f32_t *output,                                                             //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_sapphireamx_t scratch;
    // Zero once per call: each query block's first panel has correction 2^(−huge) ≈ 0 and
    // finalization re-zeroes consumed rows. Zeroing is load-bearing: the first panel's
    // correction is exp2 clamped at 2^-125 — small, but nonzero against garbage stack bits.

    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < nk_attention_chunk_sapphireamx_k_; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_sapphireamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 32);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded * sizeof(nk_bf16_t);
        nk_size_t const key_value_head_idx = head / head_group_size;
        nk_bf16_t const *keys_head_tiles = (nk_bf16_t const *)(tiles_base + tile_offsets[segment] +
                                                               key_value_head_idx * bytes_per_head);
        nk_bf16_t const *values_head_tiles = (nk_bf16_t const *)(tiles_base + tile_offsets[segment] +
                                                                 (key_value_head_count + key_value_head_idx) *
                                                                     bytes_per_head);
        nk_attention_task_sapphireamx_(queries, element_bytes, gather, output, keys_head_tiles, values_head_tiles, head,
                                       depth, position_count, position_count_padded, query_offsets[segment], row_count,
                                       query_stride_bytes, output_stride_bytes, scale2, &scratch);
    }
    _tile_release();
}

NK_API_COMPTIME void nk_attention_packed_bf16_sapphireamx(                       //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_sapphireamx_(queries, sizeof(nk_bf16_t), &nk_attention_gather_bf16_sapphireamx_,
                                     key_value_packed, output, head_count, key_value_head_count, depth, query_offsets,
                                     query_stride_bytes, output_stride_bytes, scale, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_packed_e4m3_sapphireamx(                       //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_sapphireamx_(queries, sizeof(nk_e4m3_t), &nk_attention_gather_e4m3_sapphireamx_,
                                     key_value_packed, output, head_count, key_value_head_count, depth, query_offsets,
                                     query_stride_bytes, output_stride_bytes, scale, first_task, task_count);
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_i8_sapphireamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t total_tile_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 64);
        total_tile_bytes += 2 * key_value_head_count * position_count_padded *
                            depth_padded; // K + V, one byte per element
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + total_tile_bytes;
}

NK_API_COMPTIME void nk_attention_pack_i8_sapphireamx(                                //
    nk_i8_t const *keys, nk_i8_t const *values,                                       //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t const depth_tiles = depth_padded / 64; // I8 tiles are 64 channels deep, not 32
    nk_size_t const dim_tiles = depth_padded / 16;
    nk_size_t const tile_bytes = 1024;

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 64, depth_padded);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *tile_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *tiles_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    // Output byte 4·col+q of each depth-group row comes from input byte q·16+col: one VPERMB
    // interleaves four 16-channel V rows into the quad layout TDPBUSD's B operand consumes.

    __m512i const quad_interleave_index_u8x64 = _mm512_setr_epi32( //
        0x30201000, 0x31211101, 0x32221202, 0x33231303,            //
        0x34241404, 0x35251505, 0x36261606, 0x37271707,            //
        0x38281808, 0x39291909, 0x3A2A1A0A, 0x3B2B1B0B,            //
        0x3C2C1C0C, 0x3D2D1D0D, 0x3E2E1E0E, 0x3F2F1F0F);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 64);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded;

        // K: gather 16 (strided) rows × 64 channels, transpose into the quad-interleaved
        // B-tile the same way the `dots` I8 packer does, tiles ordered [kv_tile][depth_tile].
        nk_i8_t *keys_head_tiles = (nk_i8_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                               key_value_head_idx * bytes_per_head);
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / 16; position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * 16;
            nk_size_t const valid_positions = (position_start + 16 <= position_count) ? 16
                                              : (position_start < position_count)     ? position_count - position_start
                                                                                      : 0;
            for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
                nk_size_t const channel_start = depth_tile * 64;
                nk_size_t const valid_columns = (channel_start + 64 <= depth) ? 64
                                                : (channel_start < depth)     ? depth - channel_start
                                                                              : 0;
                nk_i8_t const *source = (nk_i8_t const *)((char const *)keys +
                                                          (position_first + position_start) * key_stride_bytes) +
                                        key_value_head_idx * depth + channel_start;
                nk_dots_i8_a16x64_sapphireamx_t source_tile;
                nk_dots_i8_b64x16_sapphireamx_t transposed_tile;
                nk_dots_i8_load_a_sapphireamx_(&source_tile, source, key_stride_bytes, valid_positions, valid_columns);
                nk_dots_pack_i8_transposed_sapphireamx_(&source_tile, &transposed_tile);
                nk_i8_t *tile_output = keys_head_tiles + (position_tile_idx * depth_tiles + depth_tile) * tile_bytes;
                for (nk_size_t i = 0; i < tile_bytes; i += 64)
                    _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
            }
        }

        // V: interleave four consecutive position rows per 16-channel column group with one
        // VPERMB per quad, tiles ordered [depth_tile][position_block] (depth-major streaming for P×V).
        nk_i8_t *values_head_tiles = (nk_i8_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                                 (key_value_head_count + key_value_head_idx) * bytes_per_head);
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < dim_tiles; depth_tile_idx++) {
            nk_size_t const channel_start = depth_tile_idx * 16;
            nk_size_t const valid_columns = (channel_start + 16 <= depth) ? 16
                                            : (channel_start < depth)     ? depth - channel_start
                                                                          : 0;
            __mmask16 const columns_m16 = (__mmask16)((1u << valid_columns) - 1);
            for (nk_size_t position_block_idx = 0; position_block_idx < position_count_padded / 64;
                 position_block_idx++) {
                nk_i8_t *tile_output = values_head_tiles +
                                       (depth_tile_idx * (position_count_padded / 64) + position_block_idx) *
                                           tile_bytes;
                for (nk_size_t quad = 0; quad < 16; quad++) {
                    nk_size_t const row_first = position_block_idx * 64 + quad * 4;
                    __m512i quad_i8x64 = _mm512_setzero_si512();
                    for (nk_size_t r = 0; r < 4; r++) {
                        if (row_first + r >= position_count) continue;
                        char const *row_ptr = (char const *)values +
                                              (position_first + row_first + r) * value_stride_bytes +
                                              key_value_head_idx * depth + channel_start;
                        __m128i const row_i8x16 = _mm_maskz_loadu_epi8(columns_m16, row_ptr);
                        switch (r) { // `_mm512_inserti32x4` requires an immediate lane index
                        case 0: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 0); break;
                        case 1: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 1); break;
                        case 2: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 2); break;
                        case 3: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 3); break;
                        }
                    }
                    // Unaligned: the packed blob may live in a Python flex-array with no 64 B guarantee.
                    _mm512_storeu_si512(tile_output + quad * 64,
                                        _mm512_permutexvar_epi8(quad_interleave_index_u8x64, quad_i8x64));
                }
            }
        }
    }
    nk_compiler_barrier_sapphireamx_();
}

/** @brief Per-call I8 scratch: I32 score panel, U8 weight panel, output accumulators, packed Q tiles. */
typedef struct {
    NK_ALIGN64 nk_i32_t scores_panel[32 * nk_attention_panel_sapphireamx_k_]; // 64 KB
    NK_ALIGN64 nk_u8_t weights_panel[32 * nk_attention_panel_sapphireamx_k_]; // 16 KB
    NK_ALIGN64 nk_f32_t o_acc[nk_attention_chunk_sapphireamx_k_][32 * nk_attention_max_depth_sapphireamx_k_];
    nk_dots_i8_a16x64_sapphireamx_t q_tiles[nk_attention_chunk_sapphireamx_k_][2][4]; // 32 KB
} nk_attention_scratch_i8_sapphireamx_t_;

/**
 *  @brief Stage 1: exact-integer Q×Kᵀ for one KV panel with 2×2 register blocking.
 *
 *  Same tile schedule as the BF16 stage, but TDPBSSD over 64-channel depth steps and an
 *  I32 score panel. The row-max sweep covers only the `valid_cols` live columns: unlike
 *  the F32 path, U8-quantized weights can round to exact zero, so a zero-score padded
 *  column raising the max could zero out an all-negative row's weight sum entirely.
 */
NK_HELPER_INLINE void nk_attention_score_panel_i8_sapphireamx_( //
    nk_dots_i8_a16x64_sapphireamx_t const (*q_tiles)[4],        // [2][depth_tiles] pre-packed Q
    nk_i8_t const *keys_head_tiles, nk_size_t panel_first_tile, // K tiles, [kv_tile][depth_tile]
    nk_size_t panel_pairs, nk_size_t depth_tiles,               // panel width / 32, channels / 64
    nk_i32_t *scores_panel, nk_size_t panel_width,              // [32][panel_width]
    nk_size_t valid_channels, nk_i32_t (*panel_max)[16]) {      // live columns, [2][16] raw maxima out

    int const panel_stride_bytes = (int)(panel_width * sizeof(nk_i32_t));
    nk_size_t const k_tile_stride = depth_tiles * 1024;
    for (nk_size_t pair = 0; pair < panel_pairs; pair++) {
        nk_i8_t const *k_tile0 = keys_head_tiles + (panel_first_tile + pair * 2) * k_tile_stride;
        nk_i8_t const *k_tile1 = k_tile0 + k_tile_stride;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
            _tile_loadd(4, q_tiles[0][depth_tile].data, 64);
            _tile_loadd(5, q_tiles[1][depth_tile].data, 64);
            _tile_loadd(6, k_tile0 + depth_tile * 1024, 64);
            _tile_loadd(7, k_tile1 + depth_tile * 1024, 64);
            _tile_dpbssd(0, 4, 6);
            _tile_dpbssd(1, 4, 7);
            _tile_dpbssd(2, 5, 6);
            _tile_dpbssd(3, 5, 7);
        }
        _tile_stored(0, scores_panel + pair * 32, panel_stride_bytes);
        _tile_stored(1, scores_panel + pair * 32 + 16, panel_stride_bytes);
        _tile_stored(2, scores_panel + 16 * panel_width + pair * 32, panel_stride_bytes);
        _tile_stored(3, scores_panel + 16 * panel_width + pair * 32 + 16, panel_stride_bytes);
    }
    nk_size_t const full_cols = valid_channels & ~(nk_size_t)15;
    __mmask16 const tail_m16 = (__mmask16)((1u << (valid_channels - full_cols)) - 1);
    for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
        for (nk_size_t row = 0; row < 16; row++) {
            nk_i32_t const *scores_row = scores_panel + (row_tile_idx * 16 + row) * panel_width;
            __m512i max_i32x16 = _mm512_set1_epi32(NK_I32_MIN);
            nk_size_t channel_idx = 0;
            for (; channel_idx < full_cols; channel_idx += 16)
                max_i32x16 = _mm512_max_epi32(max_i32x16, _mm512_load_si512(scores_row + channel_idx));
            if (channel_idx < valid_channels)
                max_i32x16 = _mm512_mask_max_epi32(max_i32x16, tail_m16, max_i32x16,
                                                   _mm512_load_si512(scores_row + channel_idx));
            panel_max[row_tile_idx][row] = _mm512_reduce_max_epi32(max_i32x16);
        }
}

/**
 *  @brief I-BERT-style integer exponential for the I8 weight path: takes the base-2 argument as a
 *         Q15 fixed-point value in `[−10·2^15, 0]` and returns `round(2^t · 255)` as a U8 weight in
 *         the low byte of each I32 lane. A degree-3 fixed-point polynomial covers the fraction and a
 *         lane-variable shift applies the integer part — the AVX-512 mirror of the SME/Ice Lake helper.
 */
NK_HELPER_INLINE __m512i nk_attention_iexp2_weight_i32x16_sapphireamx_(__m512i t_q15_i32x16) {
    __m512i const whole_i32x16 = _mm512_srai_epi32(t_q15_i32x16, 15); // floor, in [-10, 0]
    __m512i const fraction_i32x16 = _mm512_and_si512(t_q15_i32x16, _mm512_set1_epi32(0x7FFF));
    __m512i poly_i32x16 = _mm512_set1_epi32(1296); // Chebyshev-fit 2^r coefficients in Q14, degree 3
    poly_i32x16 = _mm512_add_epi32(_mm512_srai_epi32(_mm512_mullo_epi32(fraction_i32x16, poly_i32x16), 15),
                                   _mm512_set1_epi32(3678));
    poly_i32x16 = _mm512_add_epi32(_mm512_srai_epi32(_mm512_mullo_epi32(fraction_i32x16, poly_i32x16), 15),
                                   _mm512_set1_epi32(11410));
    poly_i32x16 = _mm512_add_epi32(_mm512_srai_epi32(_mm512_mullo_epi32(fraction_i32x16, poly_i32x16), 15),
                                   _mm512_set1_epi32(16382));
    __m512i const scaled_i32x16 = _mm512_sub_epi32(_mm512_slli_epi32(poly_i32x16, 8), poly_i32x16); // 255 = (x<<8)-x
    __m512i const shift_i32x16 = _mm512_sub_epi32(_mm512_set1_epi32(14), whole_i32x16);
    __m512i const bias_i32x16 = _mm512_sllv_epi32(_mm512_set1_epi32(1),
                                                  _mm512_sub_epi32(_mm512_set1_epi32(13), whole_i32x16));
    return _mm512_srav_epi32(_mm512_add_epi32(scaled_i32x16, bias_i32x16), shift_i32x16);
}

/**
 *  @brief Stage 2: streaming base-2 softmax with U8 weight quantization over one panel, entirely in
 *         integer arithmetic: `w̃ = iexp2((score − m_row)·scale₂)` in Q15, with the running sum of
 *         the quantized weights accumulated in I32 and converted to F32 for the online correction.
 *         The max-scoring position lands on 255, so the sum can never be zero. Columns past
 *         `valid_channels` get zero weights (U8 zero is an exact zero weight) and no sum contribution.
 */
NK_HELPER_INLINE void nk_attention_exp_panel_i8_sapphireamx_(              //
    nk_i32_t const *scores_panel, nk_u8_t *weights_panel,                  // [32][panel] each
    nk_size_t panel_cols, nk_size_t valid_channels, nk_size_t panel_width, //
    nk_i32_t scale_fixed, nk_i32_t delta_floor, nk_i32_t const (*new_max)[16], nk_f32_t (*panel_sums)[16]) {

    __m512i const scale_fixed_i32x16 = _mm512_set1_epi32(scale_fixed);
    __m512i const delta_floor_i32x16 = _mm512_set1_epi32(delta_floor);
    nk_size_t const full_cols = valid_channels & ~(nk_size_t)15;
    __mmask16 const tail_m16 = (__mmask16)((1u << (valid_channels - full_cols)) - 1);
    for (nk_size_t row = 0; row < 32; row++) {
        nk_i32_t const *scores_row = scores_panel + row * panel_width;
        nk_u8_t *weights_row = weights_panel + row * panel_width;
        __m512i const new_max_i32x16 = _mm512_set1_epi32(new_max[row / 16][row % 16]);
        __m512i sum_i32x16 = _mm512_setzero_si512();
        nk_size_t channel_idx = 0;
        for (; channel_idx < full_cols; channel_idx += 16) {
            __m512i const delta_i32x16 = _mm512_max_epi32(
                _mm512_sub_epi32(_mm512_load_si512(scores_row + channel_idx), new_max_i32x16), delta_floor_i32x16);
            __m512i const weight_i32x16 = nk_attention_iexp2_weight_i32x16_sapphireamx_(
                _mm512_mullo_epi32(delta_i32x16, scale_fixed_i32x16));
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, weight_i32x16);
            _mm_store_si128((__m128i *)(weights_row + channel_idx), _mm512_cvtusepi32_epi8(weight_i32x16));
        }
        if (channel_idx < valid_channels) {
            __m512i const delta_i32x16 = _mm512_max_epi32(
                _mm512_sub_epi32(_mm512_load_si512(scores_row + channel_idx), new_max_i32x16), delta_floor_i32x16);
            __m512i const weight_i32x16 = _mm512_maskz_mov_epi32(
                tail_m16,
                nk_attention_iexp2_weight_i32x16_sapphireamx_(_mm512_mullo_epi32(delta_i32x16, scale_fixed_i32x16)));
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, weight_i32x16);
            _mm_store_si128((__m128i *)(weights_row + channel_idx), _mm512_cvtusepi32_epi8(weight_i32x16));
            channel_idx += 16;
        }
        for (; channel_idx < panel_cols; channel_idx += 16) // zero weights for the padded remainder
            _mm_store_si128((__m128i *)(weights_row + channel_idx), _mm_setzero_si128());
        panel_sums[row / 16][row % 16] = (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16);
    }
}

/**
 *  @brief Stage 3: P×V for one panel via TDPBUSD (U8 weights × I8 values → I32).
 *
 *  Same 2×2 schedule as the BF16 stage over 64-position depth steps: TMM0-3 stay
 *  TMM-resident across the panel, then each I32 accumulator converts to F32 once and
 *  fuses with the online correction: `o_acc = o_acc · correction + o_panel`.
 */
NK_HELPER_INLINE void nk_attention_weighted_sum_panel_i8_sapphireamx_( //
    nk_u8_t const *weights_panel, nk_size_t panel_width,               // [32][panel] U8
    nk_i8_t const *values_head_tiles, nk_size_t position_blocks,       // V tiles, [depth_tile][position_block]
    nk_size_t panel_first_step, nk_size_t panel_steps,                 // in 64-position units
    nk_size_t depth_blocks, nk_size_t output_stride_floats,            // channels / 32, o_acc row stride
    nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {              // [2][16], [32][o_stride]

    NK_ALIGN64 nk_i32_t o_panel_tile[16][16];
    int const weights_stride_bytes = (int)panel_width;
    for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_blocks; depth_tile_idx++) {
        nk_i8_t const *v_tile0 = values_head_tiles +
                                 ((depth_tile_idx * 2 + 0) * position_blocks + panel_first_step) * 1024;
        nk_i8_t const *v_tile1 = values_head_tiles +
                                 ((depth_tile_idx * 2 + 1) * position_blocks + panel_first_step) * 1024;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t step = 0; step < panel_steps; step++) {
            _tile_loadd(4, weights_panel + step * 64, weights_stride_bytes);
            _tile_loadd(5, weights_panel + 16 * panel_width + step * 64, weights_stride_bytes);
            _tile_loadd(6, v_tile0 + step * 1024, 64);
            _tile_loadd(7, v_tile1 + step * 1024, 64);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);
        }
        for (nk_size_t tile = 0; tile < 4; tile++) {
            switch (tile) { // `_tile_stored` requires an immediate tile register index
            case 0: _tile_stored(0, o_panel_tile, 64); break;
            case 1: _tile_stored(1, o_panel_tile, 64); break;
            case 2: _tile_stored(2, o_panel_tile, 64); break;
            case 3: _tile_stored(3, o_panel_tile, 64); break;
            }
            nk_size_t const row_tile_idx = tile / 2;
            nk_size_t const channel_start = depth_tile_idx * 32 + (tile % 2) * 16;
            for (nk_size_t row = 0; row < 16; row++) {
                __m512 const correction_f32x16 = _mm512_set1_ps(corrections[row_tile_idx][row]);
                __m512 accumulator_f32x16 = _mm512_load_ps(o_acc + (row_tile_idx * 16 + row) * output_stride_floats +
                                                           channel_start);
                accumulator_f32x16 = _mm512_fmadd_ps(accumulator_f32x16, correction_f32x16,
                                                     _mm512_cvtepi32_ps(_mm512_load_si512(o_panel_tile[row])));
                _mm512_store_ps(o_acc + (row_tile_idx * 16 + row) * output_stride_floats + channel_start,
                                accumulator_f32x16);
            }
        }
    }
}

/**
 *  @brief One (segment, head) I8 task: panel-flash attention with U8-quantized weights.
 *  Panel-local quantization against the running maximum differs from the serial
 *  reference's global-max quantization by design; both normalize the 255 away.
 */
NK_HELPER_INLINE void nk_attention_task_i8_sapphireamx_(                                        //
    nk_i8_t const *queries, nk_f32_t *output,                                                   //
    nk_i8_t const *keys_head_tiles, nk_i8_t const *values_head_tiles,                           //
    nk_size_t head, nk_size_t depth, nk_size_t position_count, nk_size_t position_count_padded, //
    nk_size_t query_first, nk_size_t row_count,                                                 //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale2,               //
    nk_attention_scratch_i8_sapphireamx_t_ *scratch) {

    nk_size_t const panel_width = nk_attention_panel_sapphireamx_k_;
    nk_size_t const row_blocks = nk_attention_chunk_sapphireamx_k_;
    nk_size_t const kv_padded = nk_size_round_up_to_multiple_(position_count, 64);
    nk_size_t const position_blocks = position_count_padded / 64;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t const depth_tiles = depth_padded / 64;
    nk_size_t const depth_blocks = depth_padded / 32;
    nk_size_t const output_stride_floats = depth_padded;
    nk_size_t const o_stride_out = output_stride_bytes / sizeof(nk_f32_t);

    NK_ALIGN64 nk_i32_t panel_max[2][16];
    NK_ALIGN64 nk_f32_t corrections[2][16];
    NK_ALIGN64 nk_i32_t new_max_arr[2][16];
    NK_ALIGN64 nk_f32_t panel_sums[2][16];
    __m512i row_max_i32x16[nk_attention_chunk_sapphireamx_k_][2]; // raw integer running maxima
    __m512 row_sum_f32x16[nk_attention_chunk_sapphireamx_k_][2];
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t·255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const dim_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const blocks = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                     ? (row_count - row_block_start + 31) / 32
                                     : row_blocks;
        for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const position_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                nk_size_t const valid_positions = position_start >= row_count          ? 0
                                                  : (row_count - position_start >= 16) ? 16
                                                                                       : row_count - position_start;
                for (nk_size_t depth_tile = 0; depth_tile < depth_tiles; depth_tile++) {
                    nk_size_t const channel_start = depth_tile * 64;
                    nk_size_t const valid_columns = (channel_start + 64 <= depth) ? 64 : depth - channel_start;
                    nk_dots_i8_load_a_sapphireamx_(
                        &scratch->q_tiles[row_block_idx][row_tile_idx][depth_tile],
                        (nk_i8_t const *)((char const *)queries + (query_first + position_start) * query_stride_bytes) +
                            head * depth + channel_start,
                        query_stride_bytes, valid_positions, valid_columns);
                }
                row_max_i32x16[row_block_idx][row_tile_idx] = _mm512_set1_epi32(NK_I32_MIN);
                row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_setzero_ps();
            }

        // Every query block of the chunk consumes each KV panel before the next panel is
        // touched, bounding packed-K/V re-reads to (query_count / 128) sweeps.
        for (nk_size_t panel_start = 0; panel_start < kv_padded; panel_start += panel_width) {
            nk_size_t const panel_cols = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                  : (kv_padded - panel_start);
            nk_size_t const valid_channels = (panel_start + panel_cols <= position_count)
                                                 ? panel_cols
                                                 : (position_count - panel_start);

            for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++) {
                nk_attention_score_panel_i8_sapphireamx_(
                    (nk_dots_i8_a16x64_sapphireamx_t const(*)[4])scratch->q_tiles[row_block_idx], keys_head_tiles,
                    panel_start / 16, panel_cols / 32, depth_tiles, scratch->scores_panel, panel_width, valid_channels,
                    panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512i const old_max_i32x16 = row_max_i32x16[row_block_idx][row_tile_idx];
                    __m512i const new_max_i32x16 = _mm512_max_epi32(old_max_i32x16,
                                                                    _mm512_load_si512(panel_max[row_tile_idx]));
                    __m512 const corrections_f32x16 = nk_attention_exp2_f32x16_skylake_(_mm512_mul_ps(
                        _mm512_sub_ps(_mm512_cvtepi32_ps(old_max_i32x16), _mm512_cvtepi32_ps(new_max_i32x16)),
                        _mm512_set1_ps(scale2)));
                    row_max_i32x16[row_block_idx][row_tile_idx] = new_max_i32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corrections_f32x16);
                    _mm512_store_si512(new_max_arr[row_tile_idx], new_max_i32x16);
                }
                nk_attention_exp_panel_i8_sapphireamx_(scratch->scores_panel, scratch->weights_panel, panel_cols,
                                                       valid_channels, panel_width, scale_fixed, delta_floor,
                                                       (nk_i32_t const(*)[16])new_max_arr, panel_sums);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
                    row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_fmadd_ps(
                        row_sum_f32x16[row_block_idx][row_tile_idx], _mm512_load_ps(corrections[row_tile_idx]),
                        _mm512_load_ps(panel_sums[row_tile_idx]));
                nk_attention_weighted_sum_panel_i8_sapphireamx_(
                    scratch->weights_panel, panel_width, values_head_tiles, position_blocks, panel_start / 64,
                    panel_cols / 64, depth_blocks, output_stride_floats, (nk_f32_t const(*)[16])corrections,
                    scratch->o_acc[row_block_idx]);
            }
        }

        for (nk_size_t row_block_idx = 0; row_block_idx < blocks; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const position_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                nk_size_t const valid_positions = position_start >= row_count          ? 0
                                                  : (row_count - position_start >= 16) ? 16
                                                                                       : row_count - position_start;
                if (!valid_positions) continue; // flat bound replaces the mid-nest `break`
                NK_ALIGN64 nk_f32_t row_sums[16];
                _mm512_store_ps(row_sums, row_sum_f32x16[row_block_idx][row_tile_idx]);
                for (nk_size_t row = 0; row < valid_positions; row++) {
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(1.0f / row_sums[row]);
                    nk_f32_t const *accumulator_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + position_start + row) * o_stride_out + head * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(
                            output_row + channel_idx, dim_tail_m16,
                            _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                }
                for (nk_size_t row = 0; row < 16; row++)
                    for (nk_size_t channel_idx = 0; channel_idx < output_stride_floats; channel_idx += 16)
                        _mm512_store_ps(&scratch->o_acc[row_block_idx]
                                                       [(row_tile_idx * 16 + row) * output_stride_floats + channel_idx],
                                        zero_f32x16);
            }
    }
}

NK_API_COMPTIME void nk_attention_packed_i8_sapphireamx(                         //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sapphireamx_k_) {
        nk_attention_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                      task_count);
        return;
    }

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_i8_sapphireamx_t_ scratch;
    // Zero once per call: each query block's first panel has correction 2^(−huge) ≈ 0 and
    // finalization re-zeroes consumed rows. Zeroing is load-bearing: the first panel's
    // correction is exp2 clamped at 2^-125 — small, but nonzero against garbage stack bits.

    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < nk_attention_chunk_sapphireamx_k_; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_sapphireamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 64);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded;
        nk_size_t const key_value_head_idx = head / head_group_size;
        nk_i8_t const *keys_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                           key_value_head_idx * bytes_per_head);
        nk_i8_t const *values_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                             (key_value_head_count + key_value_head_idx) *
                                                                 bytes_per_head);
        nk_attention_task_i8_sapphireamx_(queries, output, keys_head_tiles, values_head_tiles, head, depth,
                                          position_count, position_count_padded, query_offsets[segment], row_count,
                                          query_stride_bytes, output_stride_bytes, scale2, &scratch);
    }
    _tile_release();
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIREAMX
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_SAPPHIREAMX_H
