/**
 *  @brief FlashAttention-style kernels for SME.
 *  @file include/numkong/attention/sme.h
 *  @author Ash Vardanian
 *  @date January 11, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  This file implements FlashAttention-2 style scaled dot-product attention (SDPA) optimized
 *  for ARM SME instructions on Apple M4 and similar processors. The kernel computes:
 *
 *      O = softmax(Q × Kᵀ / √d) × V
 *
 *  Key features:
 *  - Online softmax: Mathematically exact, processes KV blocks incrementally
 *  - Pre-packed KV cache: BFMOPA/FMOPA-interleaved format amortizes packing for repeated inference
 *  - GQA/MQA support: Different `num_heads` and `num_kv_heads` for grouped-query attention
 *  - Pure Streaming SVE: No NEON intrinsics for non-linear operations
 *
 *  Target models (2025):
 *  - Kimi K2: `head_dim`=112, 64 heads, MHA, 128K context
 *  - LLaMA 3.1 405B: `head_dim`=128, 128 heads, 16 KV heads (GQA 8:1), 128K context
 *  - Qwen 2.5 72B: `head_dim`=128, 64 heads, 8 KV heads (GQA 8:1), 32K context
 *
 *  @section attention_sme_architecture Architecture
 *
 *  Both Q×Kᵀ and P×V phases use BFMOPA/FMOPA outer products on ZA tiles, eliminating
 *  element-wise SVE loops that dominated the original implementation. The Q matrix is
 *  pre-transposed once into a buffer matching the interleaving that ZA vertical reads
 *  would produce, so Q×Kᵀ runs as pure memory-to-BFMOPA with no per-block ZA staging.
 *
 *  Block sizes:
 *  - Bᵣ = 16 (query block rows, matches ZA32 tile height)
 *  - Bᶜ = 32 (main prefill loop, processes two KV blocks per iteration using ZA2+ZA3)
 *  - Bᶜ = 16 (tail loop for remaining KV positions, and decode path)
 *
 *  KV packing format:
 *  - K is stored in BFMOPA-interleaved format: `K_packed[kv_block][depth_step][32]` where
 *    `packed[2*ki + sub] = K[kv_block*16 + ki][2*depth_step + sub]`
 *  - V is stored in BFMOPA-interleaved format: `V_packed[kv_block][dim_tile][depth_step][32]`
 *    where `packed[2*dj + sub] = V[kv_block*16 + 2*depth_step + sub][dim_tile*16 + dj]`
 *  - The `reserved[0]` header field stores `v_dim_tile_count` for efficient V addressing
 *
 *  Softmax:
 *  - Column-wise max and exp using ZA tile vertical reads (avoids per-row horizontal extracts)
 *  - Correction skip: when the block max does not exceed the running max, the output
 *    accumulator rescaling is skipped entirely (common in later KV blocks)
 *  - Degree-3 fast exp (`nk_exp_fast_f32_sve_`) saves 1 FMA per call vs degree-4
 *  - Weights stored directly as bf16/f16 in ZA0 columns via `svzip1` (no f32 round-trip)
 *
 *  Decode path (query_len=1):
 *  - Uses element-wise SVE with scalar weight broadcasts instead of BFMOPA P×V
 *  - BFMOPA overhead too high for single-query case due to ZA setup cost
 *
 *  P×V prefill path:
 *  - 4-tile BFMOPA processing: 4 dim-tiles × 8 depth steps per KV block = 32 BFMOPA ops
 *  - ZA0-ZA3 accumulate simultaneously, read results with MOVA, add to output accumulator
 *  - Remainder dim-tiles handled 1-at-a-time using ZA0 only
 *
 *  SME tile dimensions (for SVL=512, i.e., Apple M4):
 *  - ZA32 tile: 16 × 16 `f32` elements (1KB)
 *  - `bf16`/`f16` vectors: 32 elements per SVE vector
 *
 *  @section attention_sme_history Optimization History
 *
 *  Phase 1 (January 2026): Initial implementation using ZA staging transpose for Q×Kᵀ
 *  and element-wise SVE for P×V. Q and K rows were loaded into ZA0/ZA1 horizontally,
 *  read back vertically to produce interleaved vectors for BFMOPA. The P×V phase used
 *  scalar `svmla_f32_x` loops over head_dim for each query-key pair. Softmax used
 *  degree-4 polynomial exp with per-row horizontal max/sum. Performance: ~25-50 GFLOP/s
 *  on Apple M4 (bf16, 8 heads, query_len=64, kv_len=4096, head_dim=128).
 *
 *  Phase 2 (February 2026): BFMOPA/FMOPA P×V with pre-packed V in interleaved format.
 *  Key changes integrated:
 *  - Q pre-transposed once into a buffer, eliminating per-block ZA staging for Q
 *  - K pre-packed in interleaved format, enabling pure memory-to-BFMOPA Q×Kᵀ
 *  - V pre-packed in BFMOPA-interleaved format with dim-tile blocking
 *  - P×V uses 4-tile BFMOPA accumulation (ZA0-ZA3) with pre-extracted P columns
 *  - Bᶜ=32 main loop for prefill (2 KV blocks per iteration via ZA2+ZA3)
 *  - Column-wise softmax: vertical ZA reads for max/exp instead of per-row horizontal
 *  - Correction skip when running max is unchanged
 *  - Degree-3 fast exp (~0.5% max relative error, saves 1 FMA per call)
 *  - Weights stored directly as bf16/f16 via `svzip1` (no f32 quantization round-trip)
 *  Performance: ~300-400 GFLOP/s on Apple M4 (same configuration), a 6-14× improvement.
 *
 *  Rejected approaches:
 *  - BFMOPA P×V for decode (query_len=1): ZA setup overhead exceeds element-wise SVE cost
 *  - `svdot_lane` for Q×Kᵀ: lower throughput than BFMOPA on M4
 *  - Shared ZA tiles between softmax and P×V: register pressure too high with 4-tile P×V
 */
#ifndef NK_ATTENTION_SME_H
#define NK_ATTENTION_SME_H

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

/**
 *  @brief Convert bf16 vector to f32 in registers (streaming SVE compatible).
 *
 *  BF16 is the upper 16 bits of F32, so we:
 *  1. Reinterpret bf16 as u16
 *  2. Zero-extend to u32 (unpklo for lower half)
 *  3. Shift left by 16 to place in f32 exponent+mantissa position
 *  4. Reinterpret as f32
 */
NK_INTERNAL svfloat32_t nk_bf16_to_f32_sve_(svbool_t predicate_f32x, svbfloat16_t x_bf16x) __arm_streaming {
    svuint16_t x_u16x = svreinterpret_u16_bf16(x_bf16x);
    svuint32_t x_u32x = svunpklo_u32(x_u16x);
    x_u32x = svlsl_n_u32_x(predicate_f32x, x_u32x, 16);
    return svreinterpret_f32_u32(x_u32x);
}

/**
 *  @brief Convert f32 vector to bf16 in registers with rounding (streaming SVE compatible).
 *
 *  1. Reinterpret f32 as u32
 *  2. Add rounding bias (0x8000) for round-to-nearest
 *  3. Shift right by 16
 *  4. Narrow to u16 and reinterpret as bf16
 */
NK_INTERNAL svbfloat16_t nk_f32_to_bf16_sve_(svbool_t predicate_f32x, svfloat32_t x_f32x) __arm_streaming {
    svuint32_t x_u32x = svreinterpret_u32_f32(x_f32x);
    x_u32x = svadd_n_u32_x(predicate_f32x, x_u32x, 0x8000); // Round to nearest
    x_u32x = svlsr_n_u32_x(predicate_f32x, x_u32x, 16);
    svuint16_t x_u16x = svuzp1_u16(svreinterpret_u16_u32(x_u32x), svreinterpret_u16_u32(x_u32x));
    return svreinterpret_bf16_u16(x_u16x);
}

/**
 *  @brief Packed KV cache header for attention (64-byte aligned).
 *
 *  Layout in memory:
 *  [header: 64 bytes][K tiles: variable][V tiles: variable]
 */
typedef struct {
    nk_u32_t num_kv_heads;    ///< Number of K/V heads (for GQA, may differ from Q heads)
    nk_u32_t head_dim;        ///< Original head dimension (64, 112, 128)
    nk_u32_t head_dim_padded; ///< Padded to multiple of 32 for SME
    nk_u32_t seq_len;         ///< Current sequence length
    nk_u32_t max_seq_len;     ///< Maximum sequence length (for pre-allocation)
    nk_u32_t k_offset;        ///< Byte offset to K data from header start
    nk_u32_t v_offset;        ///< Byte offset to V data from header start
    nk_u32_t reserved[9];     ///< reserved[0] = v_dim_tile_count; remainder pads to 64 bytes
} nk_attention_sme_packed_header_t;

/**
 *  @brief Fast exp approximation in Streaming SVE.
 *
 *  Uses Cody-Waite range reduction + Horner polynomial (degree 4).
 *  Accuracy: ~0.1% relative error, acceptable for softmax normalization.
 *
 *  @param pg Active predicate
 *  @param x Input vector
 *  @return exp(x) approximation
 */
NK_INTERNAL svfloat32_t nk_exp_f32_sve_(svbool_t predicate_f32x, svfloat32_t x_f32x) __arm_streaming {
    // Constants for Cody-Waite range reduction
    svfloat32_t log2e_f32x = svdup_f32(1.4426950408889634f);
    svfloat32_t ln2_hi_f32x = svdup_f32(0.693145751953125f);
    svfloat32_t ln2_lo_f32x = svdup_f32(1.42860682030941723212e-6f);

    // Clamp to avoid overflow/underflow
    svfloat32_t max_x_f32x = svdup_f32(88.3762626647949f);
    svfloat32_t min_x_f32x = svdup_f32(-87.3365447504021f);
    x_f32x = svmax_f32_m(predicate_f32x, svmin_f32_m(predicate_f32x, x_f32x, max_x_f32x), min_x_f32x);

    // n = round(x / ln(2))
    svfloat32_t n_f32x = svrintn_f32_m(svundef_f32(), predicate_f32x, svmul_f32_m(predicate_f32x, x_f32x, log2e_f32x));

    // r = x - n × ln(2) using Cody-Waite for precision
    svfloat32_t r_f32x = svmsb_f32_m(predicate_f32x, n_f32x, ln2_hi_f32x, x_f32x);
    r_f32x = svmsb_f32_m(predicate_f32x, n_f32x, ln2_lo_f32x, r_f32x);

    // Polynomial approximation for exp(r): degree 4
    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
    svfloat32_t p_f32x = svdup_f32(4.1666666667e-2f);                                  // 1/24
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(1.6666666667e-1f)); // 1/6
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(5.0000000000e-1f)); // 1/2
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(1.0f));             // 1
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(1.0f));             // 1

    // Reconstruct: exp(x) = 2ⁿ × exp(r)
    // 2ⁿ via IEEE 754 exponent manipulation
    svint32_t n_i32x = svcvt_s32_f32_m(svundef_s32(), predicate_f32x, n_f32x);
    n_i32x = svadd_s32_m(predicate_f32x, n_i32x, svdup_s32(127));
    n_i32x = svlsl_n_s32_m(predicate_f32x, n_i32x, 23);
    svfloat32_t pow2n_f32x = svreinterpret_f32_s32(n_i32x);

    return svmul_f32_m(predicate_f32x, p_f32x, pow2n_f32x);
}

/**
 *  @brief Degree-3 fast exp approximation. Max relative error ~0.5%.
 *  Saves 1 FMA per call vs degree-4 nk_exp_f32_sve_.
 */
NK_INTERNAL svfloat32_t nk_exp_fast_f32_sve_(svbool_t predicate_f32x, svfloat32_t x_f32x) __arm_streaming {
    svfloat32_t log2e_f32x = svdup_f32(1.4426950408889634f);
    svfloat32_t ln2_hi_f32x = svdup_f32(0.693145751953125f);
    svfloat32_t ln2_lo_f32x = svdup_f32(1.42860682030941723212e-6f);

    svfloat32_t max_x_f32x = svdup_f32(88.3762626647949f);
    svfloat32_t min_x_f32x = svdup_f32(-87.3365447504021f);
    x_f32x = svmax_f32_m(predicate_f32x, svmin_f32_m(predicate_f32x, x_f32x, max_x_f32x), min_x_f32x);

    svfloat32_t n_f32x = svrintn_f32_m(svundef_f32(), predicate_f32x, svmul_f32_m(predicate_f32x, x_f32x, log2e_f32x));
    svfloat32_t r_f32x = svmsb_f32_m(predicate_f32x, n_f32x, ln2_hi_f32x, x_f32x);
    r_f32x = svmsb_f32_m(predicate_f32x, n_f32x, ln2_lo_f32x, r_f32x);

    // Degree-3: exp(r) ~ 1 + r + r^2/2 + r^3/6 (drop 1/24 term)
    svfloat32_t p_f32x = svdup_f32(1.6666666667e-1f);                                  // 1/6
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(5.0000000000e-1f)); // 1/2
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(1.0f));             // 1
    p_f32x = svmad_f32_m(predicate_f32x, p_f32x, r_f32x, svdup_f32(1.0f));             // 1

    svint32_t n_i32x = svcvt_s32_f32_m(svundef_s32(), predicate_f32x, n_f32x);
    n_i32x = svadd_s32_m(predicate_f32x, n_i32x, svdup_s32(127));
    n_i32x = svlsl_n_s32_m(predicate_f32x, n_i32x, 23);
    svfloat32_t pow2n_f32x = svreinterpret_f32_s32(n_i32x);

    return svmul_f32_m(predicate_f32x, p_f32x, pow2n_f32x);
}

NK_PUBLIC nk_size_t nk_attention_packed_kv_size_bf16_sme(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_size_t max_seq_len) {
    nk_size_t head_dim_padded = (head_dim + 31) / 32 * 32;
    nk_size_t kv_blocks = (max_seq_len + 15) / 16;
    nk_size_t seq_padded = kv_blocks * 16;
    // K and V both use BFMOPA-interleaved format: [num_kv_heads, kv_blocks, depth_steps, 32]
    nk_size_t k_size = num_kv_heads * seq_padded * head_dim_padded * sizeof(nk_bf16_t);
    nk_size_t v_size = k_size;
    return sizeof(nk_attention_sme_packed_header_t) + k_size + v_size;
}

NK_PUBLIC nk_size_t nk_attention_packed_kv_size_f16_sme(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                        nk_size_t max_seq_len) {
    return nk_attention_packed_kv_size_bf16_sme(num_kv_heads, head_dim, max_seq_len);
}

__arm_locally_streaming static void nk_attention_pack_kv_bf16_sme_streaming_(nk_bf16_t const *k, nk_bf16_t const *v,
                                                                             nk_size_t num_kv_heads, nk_size_t head_dim,
                                                                             nk_size_t seq_len, nk_size_t k_stride,
                                                                             nk_size_t v_stride, void *kv_packed) {

    nk_attention_sme_packed_header_t *header = (nk_attention_sme_packed_header_t *)kv_packed;
    nk_size_t head_dim_padded = (head_dim + 31) / 32 * 32;
    nk_size_t dim_tile_count = (head_dim_padded + 15) / 16;
    nk_size_t kv_block_count = (seq_len + 15) / 16;

    nk_size_t k_depth_step_count = head_dim_padded / 2;
    nk_size_t head_elems = kv_block_count * 16 * head_dim_padded;

    header->num_kv_heads = (nk_u32_t)num_kv_heads;
    header->head_dim = (nk_u32_t)head_dim;
    header->head_dim_padded = (nk_u32_t)head_dim_padded;
    header->seq_len = (nk_u32_t)seq_len;
    header->k_offset = sizeof(nk_attention_sme_packed_header_t);
    header->reserved[0] = (nk_u32_t)dim_tile_count; // v_dim_tile_count
    header->v_offset = header->k_offset + (nk_u32_t)(num_kv_heads * head_elems * sizeof(nk_bf16_t));

    nk_bf16_t *k_packed = (nk_bf16_t *)((char *)kv_packed + header->k_offset);
    nk_bf16_t *v_packed = (nk_bf16_t *)((char *)kv_packed + header->v_offset);

    for (nk_size_t h = 0; h < num_kv_heads; h++) {
        nk_bf16_t const *k_head = k + h * k_stride;
        nk_bf16_t const *v_head = v + h * v_stride;

        // K packing: BFMOPA-interleaved format
        // K_packed[kv_block][depth_step][32] where
        //   packed[2*ki + sub] = K[kv_block*16 + ki][2*depth_step + sub]
        nk_bf16_t *k_out = k_packed + h * head_elems;
        for (nk_size_t kv_block = 0; kv_block < kv_block_count; kv_block++) {
            for (nk_size_t depth_step = 0; depth_step < k_depth_step_count; depth_step++) {
                nk_bf16_t *vec_out = k_out + (kv_block * k_depth_step_count + depth_step) * 32;
                for (nk_size_t ki = 0; ki < 16; ki++) {
                    for (nk_size_t sub = 0; sub < 2; sub++) {
                        nk_size_t row = kv_block * 16 + ki;
                        nk_size_t col = 2 * depth_step + sub;
                        nk_bf16_t zero = {0};
                        vec_out[2 * ki + sub] = (row < seq_len && col < head_dim) ? k_head[row * head_dim + col] : zero;
                    }
                }
            }
        }

        // V packing: BFMOPA-interleaved format
        nk_bf16_t *v_out = v_packed + h * head_elems;
        for (nk_size_t kv_block = 0; kv_block < kv_block_count; kv_block++) {
            for (nk_size_t dim_tile = 0; dim_tile < dim_tile_count; dim_tile++) {
                for (nk_size_t depth_step = 0; depth_step < 8; depth_step++) {
                    nk_bf16_t *vec_out = v_out + (kv_block * dim_tile_count * 8 + dim_tile * 8 + depth_step) * 32;
                    for (nk_size_t dj = 0; dj < 16; dj++) {
                        for (nk_size_t sub = 0; sub < 2; sub++) {
                            nk_size_t ki = kv_block * 16 + 2 * depth_step + sub;
                            nk_size_t d = dim_tile * 16 + dj;
                            nk_bf16_t zero = {0};
                            vec_out[2 * dj + sub] = (ki < seq_len && d < head_dim) ? v_head[ki * head_dim + d] : zero;
                        }
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_kv_bf16_sme(nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads,
                                             nk_size_t head_dim, nk_size_t seq_len, nk_size_t k_stride,
                                             nk_size_t v_stride, void *kv_packed) {
    nk_attention_pack_kv_bf16_sme_streaming_(k, v, num_kv_heads, head_dim, seq_len, k_stride, v_stride, kv_packed);
}

__arm_locally_streaming static void nk_attention_pack_kv_f16_sme_streaming_(nk_f16_t const *k, nk_f16_t const *v,
                                                                            nk_size_t num_kv_heads, nk_size_t head_dim,
                                                                            nk_size_t seq_len, nk_size_t k_stride,
                                                                            nk_size_t v_stride, void *kv_packed) {

    nk_attention_sme_packed_header_t *header = (nk_attention_sme_packed_header_t *)kv_packed;
    nk_size_t head_dim_padded = (head_dim + 31) / 32 * 32;
    nk_size_t dim_tile_count = (head_dim_padded + 15) / 16;
    nk_size_t kv_block_count = (seq_len + 15) / 16;

    nk_size_t k_depth_step_count = head_dim_padded / 2;
    nk_size_t head_elems = kv_block_count * 16 * head_dim_padded;

    header->num_kv_heads = (nk_u32_t)num_kv_heads;
    header->head_dim = (nk_u32_t)head_dim;
    header->head_dim_padded = (nk_u32_t)head_dim_padded;
    header->seq_len = (nk_u32_t)seq_len;
    header->k_offset = sizeof(nk_attention_sme_packed_header_t);
    header->reserved[0] = (nk_u32_t)dim_tile_count; // v_dim_tile_count
    header->v_offset = header->k_offset + (nk_u32_t)(num_kv_heads * head_elems * sizeof(nk_f16_t));

    nk_f16_t *k_packed = (nk_f16_t *)((char *)kv_packed + header->k_offset);
    nk_f16_t *v_packed = (nk_f16_t *)((char *)kv_packed + header->v_offset);

    for (nk_size_t h = 0; h < num_kv_heads; h++) {
        nk_f16_t const *k_head = k + h * k_stride;
        nk_f16_t const *v_head = v + h * v_stride;

        // K packing: FMOPA-interleaved format
        nk_f16_t *k_out = k_packed + h * head_elems;
        for (nk_size_t kv_block = 0; kv_block < kv_block_count; kv_block++) {
            for (nk_size_t depth_step = 0; depth_step < k_depth_step_count; depth_step++) {
                nk_f16_t *vec_out = k_out + (kv_block * k_depth_step_count + depth_step) * 32;
                for (nk_size_t ki = 0; ki < 16; ki++) {
                    for (nk_size_t sub = 0; sub < 2; sub++) {
                        nk_size_t row = kv_block * 16 + ki;
                        nk_size_t col = 2 * depth_step + sub;
                        nk_f16_t zero = {0};
                        vec_out[2 * ki + sub] = (row < seq_len && col < head_dim) ? k_head[row * head_dim + col] : zero;
                    }
                }
            }
        }

        // V packing: FMOPA-interleaved format
        nk_f16_t *v_out = v_packed + h * head_elems;
        for (nk_size_t kv_block = 0; kv_block < kv_block_count; kv_block++) {
            for (nk_size_t dim_tile = 0; dim_tile < dim_tile_count; dim_tile++) {
                for (nk_size_t depth_step = 0; depth_step < 8; depth_step++) {
                    nk_f16_t *vec_out = v_out + (kv_block * dim_tile_count * 8 + dim_tile * 8 + depth_step) * 32;
                    for (nk_size_t dj = 0; dj < 16; dj++) {
                        for (nk_size_t sub = 0; sub < 2; sub++) {
                            nk_size_t ki = kv_block * 16 + 2 * depth_step + sub;
                            nk_size_t d = dim_tile * 16 + dj;
                            nk_f16_t zero = {0};
                            vec_out[2 * dj + sub] = (ki < seq_len && d < head_dim) ? v_head[ki * head_dim + d] : zero;
                        }
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_kv_f16_sme(nk_f16_t const *k, nk_f16_t const *v, nk_size_t num_kv_heads,
                                            nk_size_t head_dim, nk_size_t seq_len, nk_size_t k_stride,
                                            nk_size_t v_stride, void *kv_packed) {
    nk_attention_pack_kv_f16_sme_streaming_(k, v, num_kv_heads, head_dim, seq_len, k_stride, v_stride, kv_packed);
}

/**
 *  @brief Optimized bf16 attention kernel with BFMOPA P×V.
 *
 *  Key design choices:
 *  - P×V uses BFMOPA with pre-packed V (4-tile accumulation) instead of element-wise SVE
 *  - Scores read via column-wise vertical ZA reads for vectorized max/exp
 *  - Weights stored directly as bf16 (no f32 round-trip)
 *  - Uses degree-3 fast exp for softmax
 *  - Correction skip when running max is unchanged
 *  - Decode path (valid_query_count==1) remains element-wise SVE (BFMOPA overhead too high)
 */
__arm_locally_streaming __arm_new("za") static void nk_attention_bf16_sme_streaming_(
    nk_bf16_t const *q,        // [query_len, head_dim]
    nk_bf16_t const *k,        // [kv_len, head_dim_padded] BFMOPA-interleaved
    nk_bf16_t const *v_packed, // BFMOPA-interleaved V for this KV head
    nk_bf16_t *output,         // [query_len, head_dim]
    nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim, nk_size_t head_dim_padded, nk_size_t dim_tile_count,
    nk_f32_t scale) {

    svbool_t const predicate_all_f32x = svptrue_b32();
    svbool_t const predicate_all_f16x = svptrue_b16();
    nk_size_t const valid_query_count = (query_len < 16) ? query_len : 16;

    svfloat32_t row_max_f32x = svdup_f32(NK_F32_MIN);
    svfloat32_t row_sum_f32x = svdup_f32(0.0f);

    NK_ALIGN64 nk_f32_t output_accumulator[16 * 256];
    svfloat32_t zero_f32x = svdup_f32(0.0f);
    for (nk_size_t i = 0; i < 16 * head_dim_padded; i += svcntw()) {
        svst1_f32(predicate_all_f32x, output_accumulator + i, zero_f32x);
    }

    nk_size_t kv_block_index = 0;
    nk_size_t kv_start = 0;
    svbool_t const batch_predicate_f32x = svwhilelt_b32(0u, 16u);

    nk_size_t const k_depth_step_count = head_dim_padded / 2;

    // Pre-transpose Q once: queries_transposed[step][16 f32 words]
    NK_ALIGN64 nk_f32_t queries_transposed[128 * 16]; // max head_dim_padded/2 * 16 = 128 * 16
    for (nk_size_t batch = 0; batch < head_dim_padded / 32; batch++) {
        svzero_mask_za(nk_sme_zero_za32_tile_0_);
        for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
            svld1_hor_za32(0, query_index, batch_predicate_f32x,
                           (nk_f32_t const *)(q + query_index * head_dim + batch * 32));
        for (nk_size_t step = 0; step < 16; step++)
            svst1_f32(predicate_all_f32x, queries_transposed + (batch * 16 + step) * 16,
                      svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, step));
    }

    // Bc=32 main loop (prefill only, skipped for decode)
    if (valid_query_count > 1) {
        for (; kv_start + 32 <= kv_len; kv_start += 32, kv_block_index += 2) {
            // Q×K^T: pure memory→BFMOPA, no ZA staging for Q or K
            svzero_mask_za(nk_sme_zero_za32_tile_2_);
            svzero_mask_za(nk_sme_zero_za32_tile_3_);
            nk_bf16_t const *keys_block_lower = k + kv_block_index * k_depth_step_count * 32;
            nk_bf16_t const *keys_block_upper = k + (kv_block_index + 1) * k_depth_step_count * 32;
            for (nk_size_t step = 0; step < k_depth_step_count; step++) {
                svbfloat16_t zn = svreinterpret_bf16_f32(svld1_f32(predicate_all_f32x, queries_transposed + step * 16));
                svbfloat16_t zm0 = svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(keys_block_lower + step * 32));
                svbfloat16_t zm1 = svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(keys_block_upper + step * 32));
                svmopa_za32_bf16_m(2, predicate_all_f32x, predicate_all_f32x, zn, zm0);
                svmopa_za32_bf16_m(3, predicate_all_f32x, predicate_all_f32x, zn, zm1);
            }

            // Pass 1: Column-wise max (read ZA2/ZA3 columns vertically)
            svfloat32_t scale_f32x = svdup_f32(scale);
            svfloat32_t block_max_f32x = svdup_f32(NK_F32_MIN);
            for (nk_size_t column_index = 0; column_index < 16; column_index++) {
                svfloat32_t score_column_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                    scale_f32x);
                block_max_f32x = svmax_f32_x(predicate_all_f32x, block_max_f32x, score_column_f32x);
            }
            for (nk_size_t column_index = 0; column_index < 16; column_index++) {
                svfloat32_t score_column_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index),
                    scale_f32x);
                block_max_f32x = svmax_f32_x(predicate_all_f32x, block_max_f32x, score_column_f32x);
            }

            // Softmax correction (fully vectorized)
            svfloat32_t new_max_f32x = svmax_f32_x(predicate_all_f32x, row_max_f32x, block_max_f32x);
            svfloat32_t correction_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, row_max_f32x, new_max_f32x));
            svbool_t max_changed = svcmplt_f32(predicate_all_f32x, correction_f32x, svdup_f32(1.0f));
            nk_u32_t max_was_updated = svptest_any(predicate_all_f32x, max_changed) ? 1 : 0;
            if (max_was_updated) row_sum_f32x = svmul_f32_x(predicate_all_f32x, row_sum_f32x, correction_f32x);
            NK_ALIGN64 nk_f32_t corrections[16];
            svst1_f32(predicate_all_f32x, corrections, correction_f32x);

            // Pass 2: Column-wise exp + fused P write + sum
            svfloat32_t sum_delta_f32x = svdup_f32(0.0f);
            svzero_mask_za(nk_sme_zero_za32_tile_0_);
            // ZA2 columns in pairs → ZA0 columns 0-7
            for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
                svfloat32_t score_even_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                    scale_f32x);
                svfloat32_t score_odd_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index + 1),
                    scale_f32x);
                svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
                svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_even_f32x);
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_odd_f32x);
                svbfloat16_t weight_pair_bf16 = svzip1_bf16(nk_f32_to_bf16_sve_(predicate_all_f32x, weight_even_f32x),
                                                            nk_f32_to_bf16_sve_(predicate_all_f32x, weight_odd_f32x));
                svwrite_ver_za32_f32_m(0, column_index / 2, predicate_all_f32x,
                                       svreinterpret_f32_bf16(weight_pair_bf16));
            }
            // ZA3 columns in pairs → ZA0 columns 8-15
            for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
                svfloat32_t score_even_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index),
                    scale_f32x);
                svfloat32_t score_odd_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index + 1),
                    scale_f32x);
                svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
                svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_even_f32x);
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_odd_f32x);
                svbfloat16_t weight_pair_bf16 = svzip1_bf16(nk_f32_to_bf16_sve_(predicate_all_f32x, weight_even_f32x),
                                                            nk_f32_to_bf16_sve_(predicate_all_f32x, weight_odd_f32x));
                svwrite_ver_za32_f32_m(0, 8 + column_index / 2, predicate_all_f32x,
                                       svreinterpret_f32_bf16(weight_pair_bf16));
            }
            row_sum_f32x = svadd_f32_x(predicate_all_f32x, row_sum_f32x, sum_delta_f32x);
            row_max_f32x = new_max_f32x;

            // Extract P columns from ZA0
            svbfloat16_t probability_column_0_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svbfloat16_t probability_column_1_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 1));
            svbfloat16_t probability_column_2_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 2));
            svbfloat16_t probability_column_3_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 3));
            svbfloat16_t probability_column_4_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 4));
            svbfloat16_t probability_column_5_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 5));
            svbfloat16_t probability_column_6_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 6));
            svbfloat16_t probability_column_7_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 7));
            svbfloat16_t probability_column_8_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 8));
            svbfloat16_t probability_column_9_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 9));
            svbfloat16_t probability_column_10_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 10));
            svbfloat16_t probability_column_11_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 11));
            svbfloat16_t probability_column_12_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 12));
            svbfloat16_t probability_column_13_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 13));
            svbfloat16_t probability_column_14_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 14));
            svbfloat16_t probability_column_15_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 15));

            // Pre-apply correction once before P×V
            svbool_t query_predicate_f16x = svwhilelt_b16_u64(0u, valid_query_count * 2);
            nk_bf16_t const *values_block_lower = v_packed + kv_block_index * dim_tile_count * 8 * 32;
            nk_bf16_t const *values_block_upper = v_packed + (kv_block_index + 1) * dim_tile_count * 8 * 32;

            if (max_was_updated) {
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svfloat32_t correction_scalar_f32x = svdup_f32(corrections[query_index]);
                    for (nk_size_t dim_offset = 0; dim_offset < head_dim_padded; dim_offset += 16)
                        svst1_f32(
                            predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_offset,
                            svmul_f32_x(predicate_all_f32x,
                                        svld1_f32(predicate_all_f32x,
                                                  output_accumulator + query_index * head_dim_padded + dim_offset),
                                        correction_scalar_f32x));
                }
            }

            // P×V: zero → BFMOPA → read → add (no ZA writes for output_accumulator)
            nk_size_t dim_tile = 0;
            for (; dim_tile + 4 <= dim_tile_count; dim_tile += 4) {
                svzero_za();
                // Block0: 8 depth steps (KV positions 0-15)
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower +
                                                                                       ((dim_tile + 3) * 8 + 7) * 32)));
                // Block1: 8 depth steps (KV positions 16-31)
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                   svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper +
                                                                                       ((dim_tile + 3) * 8 + 7) * 32)));
                // Read BFMOPA result and ADD to output_accumulator
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 1, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, query_index)));
                }
            }
            // Remainder: 1 dim_tile at a time using ZA0
            for (; dim_tile < dim_tile_count; dim_tile++) {
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_lower + (dim_tile * 8 + 7) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(values_block_upper + (dim_tile * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
                    svst1_f32(predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_tile * 16,
                              svadd_f32_x(predicate_all_f32x,
                                          svld1_f32(predicate_all_f32x,
                                                    output_accumulator + query_index * head_dim_padded + dim_tile * 16),
                                          svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
            }
        }
    }

    // Bc=16 tail loop (handles remaining KV positions and decode path)
    for (; kv_start < kv_len; kv_start += 16, kv_block_index++) {
        nk_size_t const valid_kv = ((kv_start + 16) <= kv_len) ? 16 : (kv_len - kv_start);

        // Q×K^T: pure memory→BFMOPA, no ZA staging
        svzero_mask_za(nk_sme_zero_za32_tile_2_);
        nk_bf16_t const *k_block = k + kv_block_index * k_depth_step_count * 32;
        for (nk_size_t step = 0; step < k_depth_step_count; step++) {
            svbfloat16_t zn = svreinterpret_bf16_f32(svld1_f32(predicate_all_f32x, queries_transposed + step * 16));
            svbfloat16_t zm = svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(k_block + step * 32));
            svmopa_za32_bf16_m(2, predicate_all_f32x, predicate_all_f32x, zn, zm);
        }

        // Pass 1: Column-wise max (read ZA2 columns vertically)
        svfloat32_t scale_16_f32x = svdup_f32(scale);
        svfloat32_t block_max_16_f32x = svdup_f32(NK_F32_MIN);
        for (nk_size_t column_index = 0; column_index < 16; column_index++) {
            svfloat32_t score_column_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                scale_16_f32x);
            block_max_16_f32x = svmax_f32_x(predicate_all_f32x, block_max_16_f32x, score_column_f32x);
        }

        // Softmax correction (fully vectorized)
        svfloat32_t new_max_f32x = svmax_f32_x(predicate_all_f32x, row_max_f32x, block_max_16_f32x);
        svfloat32_t correction_f32x = nk_exp_fast_f32_sve_(predicate_all_f32x,
                                                           svsub_f32_x(predicate_all_f32x, row_max_f32x, new_max_f32x));
        svbool_t max_changed_16 = svcmplt_f32(predicate_all_f32x, correction_f32x, svdup_f32(1.0f));
        nk_u32_t max_was_updated_16 = svptest_any(predicate_all_f32x, max_changed_16) ? 1 : 0;
        if (max_was_updated_16) row_sum_f32x = svmul_f32_x(predicate_all_f32x, row_sum_f32x, correction_f32x);
        NK_ALIGN64 nk_f32_t corrections[16];
        svst1_f32(predicate_all_f32x, corrections, correction_f32x);

        // Pass 2: Column-wise exp + fused P write + sum (ZA2 → ZA0 columns 0-7)
        svfloat32_t sum_delta_16_f32x = svdup_f32(0.0f);
        svzero_mask_za(nk_sme_zero_za32_tile_0_);
        for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
            svfloat32_t score_even_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                scale_16_f32x);
            svfloat32_t score_odd_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index + 1),
                scale_16_f32x);
            svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
            svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
            sum_delta_16_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_16_f32x, weight_even_f32x);
            sum_delta_16_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_16_f32x, weight_odd_f32x);
            svbfloat16_t weight_pair_bf16 = svzip1_bf16(nk_f32_to_bf16_sve_(predicate_all_f32x, weight_even_f32x),
                                                        nk_f32_to_bf16_sve_(predicate_all_f32x, weight_odd_f32x));
            svwrite_ver_za32_f32_m(0, column_index / 2, predicate_all_f32x, svreinterpret_f32_bf16(weight_pair_bf16));
        }
        row_sum_f32x = svadd_f32_x(predicate_all_f32x, row_sum_f32x, sum_delta_16_f32x);
        row_max_f32x = new_max_f32x;

        if (valid_query_count == 1) {
            // Decode path: extract f32 weights from ZA0 row 0 using SVE
            svbfloat16_t row0_bf16 = svreinterpret_bf16_f32(
                svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svbfloat16_t weights_even_bf16 = svuzp1_bf16(row0_bf16, row0_bf16);
            svbfloat16_t weights_odd_bf16 = svuzp2_bf16(row0_bf16, row0_bf16);
            NK_ALIGN64 nk_f32_t decode_weights[16];
            svst1_f32(svwhilelt_b32(0u, 8u), decode_weights,
                      nk_bf16_to_f32_sve_(svwhilelt_b32(0u, 8u), weights_even_bf16));
            svst1_f32(svwhilelt_b32(0u, 8u), decode_weights + 8,
                      nk_bf16_to_f32_sve_(svwhilelt_b32(0u, 8u), weights_odd_bf16));
            NK_ALIGN64 nk_f32_t decode_weights_ordered[16];
            for (nk_size_t i = 0; i < 8; i++) {
                decode_weights_ordered[2 * i] = decode_weights[i];
                decode_weights_ordered[2 * i + 1] = decode_weights[8 + i];
            }
            svfloat32_t corr_f32x = svdup_f32(corrections[0]);
            for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(d, head_dim);
                svfloat32_t acc_f32x = svmul_f32_x(predicate_f32x, svld1_f32(predicate_f32x, output_accumulator + d),
                                                   corr_f32x);
                for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                    nk_size_t dim_tile = d / 16, depth_s = ki / 2, sub = ki % 2;
                    nk_bf16_t const *v_vec = v_packed +
                                             (kv_block_index * dim_tile_count * 8 + dim_tile * 8 + depth_s) * 32;
                    svbfloat16_t packed_bf16x = svld1_bf16(predicate_all_f16x, (bfloat16_t const *)v_vec);
                    svbfloat16_t v_selected = (sub == 0) ? svuzp1_bf16(packed_bf16x, packed_bf16x)
                                                         : svuzp2_bf16(packed_bf16x, packed_bf16x);
                    acc_f32x = svmla_f32_x(predicate_f32x, acc_f32x, svdup_f32(decode_weights_ordered[ki]),
                                           nk_bf16_to_f32_sve_(predicate_f32x, v_selected));
                }
                svst1_f32(predicate_f32x, output_accumulator + d, acc_f32x);
            }
        }
        else {
            // Prefill Bc=16: extract P columns, pre-apply correction, add-after P×V
            svbool_t query_predicate_f16x = svwhilelt_b16_u64(0u, valid_query_count * 2);

            svbfloat16_t probability_column_0_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svbfloat16_t probability_column_1_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 1));
            svbfloat16_t probability_column_2_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 2));
            svbfloat16_t probability_column_3_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 3));
            svbfloat16_t probability_column_4_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 4));
            svbfloat16_t probability_column_5_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 5));
            svbfloat16_t probability_column_6_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 6));
            svbfloat16_t probability_column_7_f32x = svreinterpret_bf16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 7));

            nk_bf16_t const *v_block = v_packed + kv_block_index * dim_tile_count * 8 * 32;

            // Pre-apply correction
            if (max_was_updated_16) {
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svfloat32_t correction_scalar_f32x = svdup_f32(corrections[query_index]);
                    for (nk_size_t dim_offset = 0; dim_offset < head_dim_padded; dim_offset += 16)
                        svst1_f32(
                            predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_offset,
                            svmul_f32_x(predicate_all_f32x,
                                        svld1_f32(predicate_all_f32x,
                                                  output_accumulator + query_index * head_dim_padded + dim_offset),
                                        correction_scalar_f32x));
                }
            }

            // P×V: zero → BFMOPA → read → add
            nk_size_t dim_tile = 0;
            for (; dim_tile + 4 <= dim_tile_count; dim_tile += 4) {
                svzero_za();
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_bf16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + ((dim_tile + 3) * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 1, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, query_index)));
                }
            }
            for (; dim_tile < dim_tile_count; dim_tile++) {
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_bf16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_bf16(predicate_all_f16x, (bfloat16_t const *)(v_block + (dim_tile * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
                    svst1_f32(predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_tile * 16,
                              svadd_f32_x(predicate_all_f32x,
                                          svld1_f32(predicate_all_f32x,
                                                    output_accumulator + query_index * head_dim_padded + dim_tile * 16),
                                          svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
            }
        }
    }

    // Final normalization
    NK_ALIGN64 nk_f32_t final_sums[16];
    svst1_f32(predicate_all_f32x, final_sums, row_sum_f32x);

    for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
        nk_f32_t inv_sum = (final_sums[query_index] > 0.0f) ? (1.0f / final_sums[query_index]) : 0.0f;
        svfloat32_t inv_sum_f32x = svdup_f32(inv_sum);

        for (nk_size_t dim_offset = 0; dim_offset < head_dim; dim_offset += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(dim_offset, head_dim);
            svfloat32_t output_f32x = svmul_f32_x(
                predicate_f32x,
                svld1_f32(predicate_f32x, output_accumulator + query_index * head_dim_padded + dim_offset),
                inv_sum_f32x);
            svbfloat16_t output_bf16x = nk_f32_to_bf16_sve_(predicate_f32x, output_f32x);
            nk_size_t store_count = (head_dim - dim_offset) < (nk_size_t)svcntw() ? (head_dim - dim_offset)
                                                                                  : (nk_size_t)svcntw();
            svbool_t store_predicate_f16x = svwhilelt_b16_u64(0u, store_count);
            svst1_bf16(store_predicate_f16x, (bfloat16_t *)(output + query_index * head_dim + dim_offset),
                       output_bf16x);
        }
    }
}

NK_PUBLIC void nk_attention_bf16_sme(nk_bf16_t const *q, void const *kv_packed, nk_bf16_t *output, nk_size_t num_heads,
                                     nk_size_t num_kv_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim,
                                     nk_f32_t scale) {

    nk_attention_sme_packed_header_t const *header = (nk_attention_sme_packed_header_t const *)kv_packed;
    nk_size_t head_dim_padded = header->head_dim_padded;
    nk_size_t dim_tile_count = header->reserved[0]; // v_dim_tile_count
    nk_size_t kv_blocks = (kv_len + 15) / 16;
    nk_size_t kv_head_stride = kv_blocks * 16 * head_dim_padded;

    nk_bf16_t const *k_packed = (nk_bf16_t const *)((char const *)kv_packed + header->k_offset);
    nk_bf16_t const *v_packed = (nk_bf16_t const *)((char const *)kv_packed + header->v_offset);

    nk_size_t group_size = (num_kv_heads > 0) ? num_heads / num_kv_heads : 1;

    for (nk_size_t q_head = 0; q_head < num_heads; q_head++) {
        nk_size_t kv_head = q_head / group_size;

        nk_bf16_t const *q_ptr = q + q_head * query_len * head_dim;
        nk_bf16_t const *k_ptr = k_packed + kv_head * kv_head_stride;
        nk_bf16_t const *v_ptr = v_packed + kv_head * kv_head_stride;
        nk_bf16_t *out_ptr = output + q_head * query_len * head_dim;

        for (nk_size_t q_start = 0; q_start < query_len; q_start += 16) {
            nk_size_t q_block_len = (q_start + 16 < query_len) ? 16 : (query_len - q_start);

            nk_attention_bf16_sme_streaming_(q_ptr + q_start * head_dim, k_ptr, v_ptr, out_ptr + q_start * head_dim,
                                             q_block_len, kv_len, head_dim, head_dim_padded, dim_tile_count, scale);
        }
    }
}

__arm_locally_streaming __arm_new("za") static void nk_attention_f16_sme_streaming_(
    nk_f16_t const *q,        // [query_len, head_dim]
    nk_f16_t const *k,        // [kv_len, head_dim_padded] FMOPA-interleaved
    nk_f16_t const *v_packed, // FMOPA-interleaved V for this KV head
    nk_f16_t *output,         // [query_len, head_dim]
    nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim, nk_size_t head_dim_padded, nk_size_t dim_tile_count,
    nk_f32_t scale) {

    svbool_t const predicate_all_f32x = svptrue_b32();
    svbool_t const predicate_all_f16x = svptrue_b16();
    nk_size_t const valid_query_count = (query_len < 16) ? query_len : 16;

    NK_ALIGN64 nk_f32_t row_max[16];
    NK_ALIGN64 nk_f32_t row_sum[16];
    NK_ALIGN64 nk_f32_t output_accumulator[16 * 256];

    svst1_f32(predicate_all_f32x, row_max, svdup_f32(NK_F32_MIN));
    svst1_f32(predicate_all_f32x, row_sum, svdup_f32(0.0f));
    svfloat32_t zero_f32x = svdup_f32(0.0f);
    for (nk_size_t i = 0; i < 16 * head_dim_padded; i += svcntw()) {
        svst1_f32(predicate_all_f32x, output_accumulator + i, zero_f32x);
    }

    nk_size_t kv_block_index = 0;
    nk_size_t kv_start = 0;
    svbool_t const batch_predicate_f32x = svwhilelt_b32(0u, 16u);

    nk_size_t const k_depth_step_count = head_dim_padded / 2;

    // Pre-transpose Q once: queries_transposed[step][16 f32 words]
    // queries_transposed[step] reinterpret-as-f16 = {Q[0][2s], Q[0][2s+1], Q[1][2s], Q[1][2s+1], ...}
    // This is the same interleaving ZA0 vertical reads would produce.
    NK_ALIGN64 nk_f32_t queries_transposed[128 * 16]; // max head_dim_padded/2 * 16 = 128 * 16
    for (nk_size_t batch = 0; batch < head_dim_padded / 32; batch++) {
        svzero_mask_za(nk_sme_zero_za32_tile_0_);
        for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
            svld1_hor_za32(0, query_index, batch_predicate_f32x,
                           (nk_f32_t const *)(q + query_index * head_dim + batch * 32));
        for (nk_size_t step = 0; step < 16; step++)
            svst1_f32(predicate_all_f32x, queries_transposed + (batch * 16 + step) * 16,
                      svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, step));
    }

    // === Bc=32 main loop (prefill only, skipped for decode) ===
    if (valid_query_count > 1) {
        for (; kv_start + 32 <= kv_len; kv_start += 32, kv_block_index += 2) {
            // Q×K^T: pure memory→FMOPA, no ZA staging for Q or K
            svzero_mask_za(nk_sme_zero_za32_tile_2_);
            svzero_mask_za(nk_sme_zero_za32_tile_3_);
            nk_f16_t const *keys_block_lower = k + kv_block_index * k_depth_step_count * 32;
            nk_f16_t const *keys_block_upper = k + (kv_block_index + 1) * k_depth_step_count * 32;
            for (nk_size_t step = 0; step < k_depth_step_count; step++) {
                svfloat16_t zn = svreinterpret_f16_f32(svld1_f32(predicate_all_f32x, queries_transposed + step * 16));
                svfloat16_t zm0 = svld1_f16(predicate_all_f16x, (float16_t const *)(keys_block_lower + step * 32));
                svfloat16_t zm1 = svld1_f16(predicate_all_f16x, (float16_t const *)(keys_block_upper + step * 32));
                svmopa_za32_f16_m(2, predicate_all_f32x, predicate_all_f32x, zn, zm0);
                svmopa_za32_f16_m(3, predicate_all_f32x, predicate_all_f32x, zn, zm1);
            }
            // ZA2 = scores[query_index][0:15], ZA3 = scores[query_index][16:31]

            // Pass 1: Column-wise max (read ZA2/ZA3 columns vertically)
            svfloat32_t scale_f32x = svdup_f32(scale);
            svfloat32_t block_max_f32x = svdup_f32(NK_F32_MIN);
            for (nk_size_t column_index = 0; column_index < 16; column_index++) {
                svfloat32_t score_column_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                    scale_f32x);
                block_max_f32x = svmax_f32_x(predicate_all_f32x, block_max_f32x, score_column_f32x);
            }
            for (nk_size_t column_index = 0; column_index < 16; column_index++) {
                svfloat32_t score_column_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index),
                    scale_f32x);
                block_max_f32x = svmax_f32_x(predicate_all_f32x, block_max_f32x, score_column_f32x);
            }

            // Softmax correction (vectorized via array load/store)
            svfloat32_t old_max_f32x = svld1_f32(predicate_all_f32x, row_max);
            svfloat32_t new_max_f32x = svmax_f32_x(predicate_all_f32x, old_max_f32x, block_max_f32x);
            svfloat32_t correction_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, old_max_f32x, new_max_f32x));
            svbool_t max_changed = svcmplt_f32(predicate_all_f32x, correction_f32x, svdup_f32(1.0f));
            nk_u32_t max_was_updated = svptest_any(predicate_all_f32x, max_changed) ? 1 : 0;
            svfloat32_t row_sum_corrected_f32x = svld1_f32(predicate_all_f32x, row_sum);
            if (max_was_updated)
                row_sum_corrected_f32x = svmul_f32_x(predicate_all_f32x, row_sum_corrected_f32x, correction_f32x);
            NK_ALIGN64 nk_f32_t corrections[16];
            svst1_f32(predicate_all_f32x, corrections, correction_f32x);

            // Pass 2: Column-wise exp + fused P write + sum
            svfloat32_t sum_delta_f32x = svdup_f32(0.0f);
            svzero_mask_za(nk_sme_zero_za32_tile_0_);
            // ZA2 columns in pairs -> ZA0 columns 0-7
            for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
                svfloat32_t score_even_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                    scale_f32x);
                svfloat32_t score_odd_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index + 1),
                    scale_f32x);
                svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
                svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_even_f32x);
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_odd_f32x);
                svfloat16_t weight_pair_f16x = svzip1_f16(svcvt_f16_f32_x(predicate_all_f32x, weight_even_f32x),
                                                          svcvt_f16_f32_x(predicate_all_f32x, weight_odd_f32x));
                svwrite_ver_za32_f32_m(0, column_index / 2, predicate_all_f32x,
                                       svreinterpret_f32_f16(weight_pair_f16x));
            }
            // ZA3 columns in pairs -> ZA0 columns 8-15
            for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
                svfloat32_t score_even_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index),
                    scale_f32x);
                svfloat32_t score_odd_f32x = svmul_f32_x(
                    predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, column_index + 1),
                    scale_f32x);
                svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
                svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                    predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_even_f32x);
                sum_delta_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_f32x, weight_odd_f32x);
                svfloat16_t weight_pair_f16x = svzip1_f16(svcvt_f16_f32_x(predicate_all_f32x, weight_even_f32x),
                                                          svcvt_f16_f32_x(predicate_all_f32x, weight_odd_f32x));
                svwrite_ver_za32_f32_m(0, 8 + column_index / 2, predicate_all_f32x,
                                       svreinterpret_f32_f16(weight_pair_f16x));
            }
            row_sum_corrected_f32x = svadd_f32_x(predicate_all_f32x, row_sum_corrected_f32x, sum_delta_f32x);
            svst1_f32(predicate_all_f32x, row_sum, row_sum_corrected_f32x);
            svst1_f32(predicate_all_f32x, row_max, new_max_f32x);

            // Extract P columns from ZA0
            svfloat16_t probability_column_0_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svfloat16_t probability_column_1_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 1));
            svfloat16_t probability_column_2_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 2));
            svfloat16_t probability_column_3_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 3));
            svfloat16_t probability_column_4_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 4));
            svfloat16_t probability_column_5_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 5));
            svfloat16_t probability_column_6_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 6));
            svfloat16_t probability_column_7_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 7));
            svfloat16_t probability_column_8_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 8));
            svfloat16_t probability_column_9_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 9));
            svfloat16_t probability_column_10_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 10));
            svfloat16_t probability_column_11_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 11));
            svfloat16_t probability_column_12_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 12));
            svfloat16_t probability_column_13_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 13));
            svfloat16_t probability_column_14_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 14));
            svfloat16_t probability_column_15_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 15));

            // Pre-apply correction once before P×V
            svbool_t query_predicate_f16x = svwhilelt_b16_u64(0u, valid_query_count * 2);
            nk_f16_t const *values_block_lower = v_packed + kv_block_index * dim_tile_count * 8 * 32;
            nk_f16_t const *values_block_upper = v_packed + (kv_block_index + 1) * dim_tile_count * 8 * 32;

            if (max_was_updated) {
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svfloat32_t correction_scalar_f32x = svdup_f32(corrections[query_index]);
                    for (nk_size_t dim_offset = 0; dim_offset < head_dim_padded; dim_offset += 16)
                        svst1_f32(
                            predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_offset,
                            svmul_f32_x(predicate_all_f32x,
                                        svld1_f32(predicate_all_f32x,
                                                  output_accumulator + query_index * head_dim_padded + dim_offset),
                                        correction_scalar_f32x));
                }
            }

            // P×V: zero -> FMOPA -> read -> add (no ZA writes for output_accumulator)
            nk_size_t dim_tile = 0;
            for (; dim_tile + 4 <= dim_tile_count; dim_tile += 4) {
                svzero_za();
                // Block0: 8 depth steps (KV positions 0-15)
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_lower + ((dim_tile + 3) * 8 + 7) * 32)));
                // Block1: 8 depth steps (KV positions 16-31)
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_f16_m(0, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_f16_m(1, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_f16_m(2, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_f16_m(3, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                                  svld1_f16(predicate_all_f16x,
                                            (float16_t const *)(values_block_upper + ((dim_tile + 3) * 8 + 7) * 32)));
                // Read FMOPA result and ADD to output_accumulator
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 1, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, query_index)));
                }
            }
            // Remainder: 1 dim_tile at a time using ZA0
            for (; dim_tile < dim_tile_count; dim_tile++) {
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_lower + (dim_tile * 8 + 7) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_8_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_9_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_10_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_11_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_12_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_13_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_14_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_15_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(values_block_upper + (dim_tile * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
                    svst1_f32(predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_tile * 16,
                              svadd_f32_x(predicate_all_f32x,
                                          svld1_f32(predicate_all_f32x,
                                                    output_accumulator + query_index * head_dim_padded + dim_tile * 16),
                                          svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
            }
        }
    }

    // === Bc=16 tail loop (handles remaining KV positions and decode path) ===
    for (; kv_start < kv_len; kv_start += 16, kv_block_index++) {
        nk_size_t const valid_kv = ((kv_start + 16) <= kv_len) ? 16 : (kv_len - kv_start);

        // Q×K^T: pure memory→FMOPA, no ZA staging
        svzero_mask_za(nk_sme_zero_za32_tile_2_);
        nk_f16_t const *k_block = k + kv_block_index * k_depth_step_count * 32;
        for (nk_size_t step = 0; step < k_depth_step_count; step++) {
            svfloat16_t zn = svreinterpret_f16_f32(svld1_f32(predicate_all_f32x, queries_transposed + step * 16));
            svfloat16_t zm = svld1_f16(predicate_all_f16x, (float16_t const *)(k_block + step * 32));
            svmopa_za32_f16_m(2, predicate_all_f32x, predicate_all_f32x, zn, zm);
        }

        // Pass 1: Column-wise max (read ZA2 columns vertically)
        svfloat32_t scale_16_f32x = svdup_f32(scale);
        svfloat32_t block_max_16_f32x = svdup_f32(NK_F32_MIN);
        for (nk_size_t column_index = 0; column_index < 16; column_index++) {
            svfloat32_t score_column_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                scale_16_f32x);
            block_max_16_f32x = svmax_f32_x(predicate_all_f32x, block_max_16_f32x, score_column_f32x);
        }

        svfloat32_t old_max_f32x = svld1_f32(predicate_all_f32x, row_max);
        svfloat32_t new_max_f32x = svmax_f32_x(predicate_all_f32x, old_max_f32x, block_max_16_f32x);
        svfloat32_t correction_f32x = nk_exp_fast_f32_sve_(predicate_all_f32x,
                                                           svsub_f32_x(predicate_all_f32x, old_max_f32x, new_max_f32x));
        svbool_t max_changed_16 = svcmplt_f32(predicate_all_f32x, correction_f32x, svdup_f32(1.0f));
        nk_u32_t max_was_updated_16 = svptest_any(predicate_all_f32x, max_changed_16) ? 1 : 0;
        svfloat32_t row_sum_corrected_f32x = svld1_f32(predicate_all_f32x, row_sum);
        if (max_was_updated_16)
            row_sum_corrected_f32x = svmul_f32_x(predicate_all_f32x, row_sum_corrected_f32x, correction_f32x);
        NK_ALIGN64 nk_f32_t corrections[16];
        svst1_f32(predicate_all_f32x, corrections, correction_f32x);

        // Pass 2: Column-wise exp + fused P write + sum (ZA2 → ZA0 columns 0-7)
        svfloat32_t sum_delta_16_f32x = svdup_f32(0.0f);
        svzero_mask_za(nk_sme_zero_za32_tile_0_);
        for (nk_size_t column_index = 0; column_index < 16; column_index += 2) {
            svfloat32_t score_even_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index),
                scale_16_f32x);
            svfloat32_t score_odd_f32x = svmul_f32_x(
                predicate_all_f32x, svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, column_index + 1),
                scale_16_f32x);
            svfloat32_t weight_even_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_even_f32x, new_max_f32x));
            svfloat32_t weight_odd_f32x = nk_exp_fast_f32_sve_(
                predicate_all_f32x, svsub_f32_x(predicate_all_f32x, score_odd_f32x, new_max_f32x));
            sum_delta_16_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_16_f32x, weight_even_f32x);
            sum_delta_16_f32x = svadd_f32_x(predicate_all_f32x, sum_delta_16_f32x, weight_odd_f32x);
            svfloat16_t weight_pair_f16x = svzip1_f16(svcvt_f16_f32_x(predicate_all_f32x, weight_even_f32x),
                                                      svcvt_f16_f32_x(predicate_all_f32x, weight_odd_f32x));
            svwrite_ver_za32_f32_m(0, column_index / 2, predicate_all_f32x, svreinterpret_f32_f16(weight_pair_f16x));
        }
        row_sum_corrected_f32x = svadd_f32_x(predicate_all_f32x, row_sum_corrected_f32x, sum_delta_16_f32x);
        svst1_f32(predicate_all_f32x, row_sum, row_sum_corrected_f32x);
        svst1_f32(predicate_all_f32x, row_max, new_max_f32x);

        if (valid_query_count == 1) {
            // Decode path: extract f32 weights from ZA0 row 0 using SVE
            svfloat16_t row0_f16 = svreinterpret_f16_f32(svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svfloat16_t weights_even_f16 = svuzp1_f16(row0_f16, row0_f16);
            svfloat16_t weights_odd_f16 = svuzp2_f16(row0_f16, row0_f16);
            NK_ALIGN64 nk_f32_t decode_weights[16];
            svst1_f32(svwhilelt_b32(0u, 8u), decode_weights, svcvt_f32_f16_x(svwhilelt_b32(0u, 8u), weights_even_f16));
            svst1_f32(svwhilelt_b32(0u, 8u), decode_weights + 8,
                      svcvt_f32_f16_x(svwhilelt_b32(0u, 8u), weights_odd_f16));
            NK_ALIGN64 nk_f32_t decode_weights_ordered[16];
            for (nk_size_t i = 0; i < 8; i++) {
                decode_weights_ordered[2 * i] = decode_weights[i];
                decode_weights_ordered[2 * i + 1] = decode_weights[8 + i];
            }
            svfloat32_t corr_f32x = svdup_f32(corrections[0]);
            for (nk_size_t d = 0; d < head_dim; d += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(d, head_dim);
                svfloat32_t acc_f32x = svmul_f32_x(predicate_f32x, svld1_f32(predicate_f32x, output_accumulator + d),
                                                   corr_f32x);
                for (nk_size_t ki = 0; ki < valid_kv; ki++) {
                    nk_size_t dim_tile = d / 16, depth_s = ki / 2, sub = ki % 2;
                    nk_f16_t const *v_vec = v_packed +
                                            (kv_block_index * dim_tile_count * 8 + dim_tile * 8 + depth_s) * 32;
                    svfloat16_t packed_f16x = svld1_f16(predicate_all_f16x, (float16_t const *)v_vec);
                    svfloat16_t v_selected = (sub == 0) ? svuzp1_f16(packed_f16x, packed_f16x)
                                                        : svuzp2_f16(packed_f16x, packed_f16x);
                    acc_f32x = svmla_f32_x(predicate_f32x, acc_f32x, svdup_f32(decode_weights_ordered[ki]),
                                           svcvt_f32_f16_x(predicate_f32x, v_selected));
                }
                svst1_f32(predicate_f32x, output_accumulator + d, acc_f32x);
            }
        }
        else {
            // Prefill Bc=16: extract P columns, pre-apply correction, add-after P×V
            svbool_t query_predicate_f16x = svwhilelt_b16_u64(0u, valid_query_count * 2);

            svfloat16_t probability_column_0_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 0));
            svfloat16_t probability_column_1_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 1));
            svfloat16_t probability_column_2_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 2));
            svfloat16_t probability_column_3_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 3));
            svfloat16_t probability_column_4_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 4));
            svfloat16_t probability_column_5_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 5));
            svfloat16_t probability_column_6_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 6));
            svfloat16_t probability_column_7_f32x = svreinterpret_f16_f32(
                svread_ver_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, 7));

            nk_f16_t const *v_block = v_packed + kv_block_index * dim_tile_count * 8 * 32;

            if (max_was_updated_16) {
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svfloat32_t correction_scalar_f32x = svdup_f32(corrections[query_index]);
                    for (nk_size_t dim_offset = 0; dim_offset < head_dim_padded; dim_offset += 16)
                        svst1_f32(
                            predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_offset,
                            svmul_f32_x(predicate_all_f32x,
                                        svld1_f32(predicate_all_f32x,
                                                  output_accumulator + query_index * head_dim_padded + dim_offset),
                                        correction_scalar_f32x));
                }
            }

            nk_size_t dim_tile = 0;
            for (; dim_tile + 4 <= dim_tile_count; dim_tile += 4) {
                svzero_za();
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 0) * 8 + 7) * 32)));
                svmopa_za32_f16_m(
                    1, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 1) * 8 + 7) * 32)));
                svmopa_za32_f16_m(
                    2, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 2) * 8 + 7) * 32)));
                svmopa_za32_f16_m(
                    3, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + ((dim_tile + 3) * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 0) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 1) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 1, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 2) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 2, query_index)));
                    svst1_f32(
                        predicate_all_f32x, output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16,
                        svadd_f32_x(predicate_all_f32x,
                                    svld1_f32(predicate_all_f32x,
                                              output_accumulator + query_index * head_dim_padded + (dim_tile + 3) * 16),
                                    svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 3, query_index)));
                }
            }
            for (; dim_tile < dim_tile_count; dim_tile++) {
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_0_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 0) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_1_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 1) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_2_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 2) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_3_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 3) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_4_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 4) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_5_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 5) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_6_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 6) * 32)));
                svmopa_za32_f16_m(
                    0, query_predicate_f16x, predicate_all_f16x, probability_column_7_f32x,
                    svld1_f16(predicate_all_f16x, (float16_t const *)(v_block + (dim_tile * 8 + 7) * 32)));
                for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++)
                    svst1_f32(predicate_all_f32x, output_accumulator + query_index * head_dim_padded + dim_tile * 16,
                              svadd_f32_x(predicate_all_f32x,
                                          svld1_f32(predicate_all_f32x,
                                                    output_accumulator + query_index * head_dim_padded + dim_tile * 16),
                                          svread_hor_za32_f32_m(svdup_f32(0), predicate_all_f32x, 0, query_index)));
            }
        }
    }

    // Final normalization
    svfloat32_t final_sum_f32x = svld1_f32(predicate_all_f32x, row_sum);
    svfloat32_t ones_f32x = svdup_f32(1.0f);
    svfloat32_t zeros_f32x = svdup_f32(0.0f);
    svbool_t sum_positive = svcmpgt_f32(predicate_all_f32x, final_sum_f32x, zeros_f32x);
    svfloat32_t inv_sum_f32x = svsel_f32(sum_positive, svdiv_f32_x(predicate_all_f32x, ones_f32x, final_sum_f32x),
                                         zeros_f32x);

    NK_ALIGN64 nk_f32_t inv_sums[16];
    svst1_f32(predicate_all_f32x, inv_sums, inv_sum_f32x);

    for (nk_size_t query_index = 0; query_index < valid_query_count; query_index++) {
        svfloat32_t inv_sum_f32x = svdup_f32(inv_sums[query_index]);
        for (nk_size_t dim_offset = 0; dim_offset < head_dim; dim_offset += svcntw()) {
            svbool_t predicate_f32x = svwhilelt_b32_u64(dim_offset, head_dim);
            svfloat32_t output_f32x = svmul_f32_x(
                predicate_f32x,
                svld1_f32(predicate_f32x, output_accumulator + query_index * head_dim_padded + dim_offset),
                inv_sum_f32x);
            svfloat16_t output_f16x = svcvt_f16_f32_x(predicate_f32x, output_f32x);
            nk_size_t store_count = (head_dim - dim_offset) < (nk_size_t)svcntw() ? (head_dim - dim_offset)
                                                                                  : (nk_size_t)svcntw();
            svbool_t predicate_f16x = svwhilelt_b16_u64(0u, store_count);
            svst1_f16(predicate_f16x, (float16_t *)(output + query_index * head_dim + dim_offset), output_f16x);
        }
    }
}

NK_PUBLIC void nk_attention_f16_sme(nk_f16_t const *q, void const *kv_packed, nk_f16_t *output, nk_size_t num_heads,
                                    nk_size_t num_kv_heads, nk_size_t query_len, nk_size_t kv_len, nk_size_t head_dim,
                                    nk_f32_t scale) {

    nk_attention_sme_packed_header_t const *header = (nk_attention_sme_packed_header_t const *)kv_packed;
    nk_size_t head_dim_padded = header->head_dim_padded;
    nk_size_t dim_tile_count = header->reserved[0];
    nk_size_t kv_blocks = (kv_len + 15) / 16;
    // K and V both use interleaved format: kv_blocks * 16 * head_dim_padded elements per head
    nk_size_t kv_head_stride = kv_blocks * 16 * head_dim_padded;

    nk_f16_t const *k_packed = (nk_f16_t const *)((char const *)kv_packed + header->k_offset);
    nk_f16_t const *v_packed = (nk_f16_t const *)((char const *)kv_packed + header->v_offset);

    nk_size_t group_size = (num_kv_heads > 0) ? num_heads / num_kv_heads : 1;

    for (nk_size_t q_head = 0; q_head < num_heads; q_head++) {
        nk_size_t kv_head = q_head / group_size;

        nk_f16_t const *q_ptr = q + q_head * query_len * head_dim;
        nk_f16_t const *k_ptr = k_packed + kv_head * kv_head_stride;
        nk_f16_t const *v_ptr = v_packed + kv_head * kv_head_stride;
        nk_f16_t *out_ptr = output + q_head * query_len * head_dim;

        for (nk_size_t q_start = 0; q_start < query_len; q_start += 16) {
            nk_size_t q_block_len = (q_start + 16 < query_len) ? 16 : (query_len - q_start);

            nk_attention_f16_sme_streaming_(q_ptr + q_start * head_dim, k_ptr, v_ptr, out_ptr + q_start * head_dim,
                                            q_block_len, kv_len, head_dim, head_dim_padded, dim_tile_count, scale);
        }
    }
}

NK_PUBLIC void nk_attention_causal_bf16_sme(nk_bf16_t const *q, void const *kv_packed, nk_bf16_t *output,
                                            nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t query_len,
                                            nk_size_t kv_len, nk_size_t head_dim, nk_f32_t scale) {
    // TODO: Implement proper causal masking with block skipping
    // For now, delegate to full attention (correct for decode where query_len=1)
    nk_attention_bf16_sme(q, kv_packed, output, num_heads, num_kv_heads, query_len, kv_len, head_dim, scale);
}

NK_PUBLIC void nk_attention_causal_f16_sme(nk_f16_t const *q, void const *kv_packed, nk_f16_t *output,
                                           nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t query_len,
                                           nk_size_t kv_len, nk_size_t head_dim, nk_f32_t scale) {
    // TODO: Implement proper causal masking with block skipping
    // For now, delegate to full attention (correct for decode where query_len=1)
    nk_attention_f16_sme(q, kv_packed, output, num_heads, num_kv_heads, query_len, kv_len, head_dim, scale);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_ATTENTION_SME_H
