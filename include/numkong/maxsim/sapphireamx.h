/**
 *  @file include/numkong/maxsim/sapphireamx.h
 *  @author Ash Vardanian
 *  @date March 7, 2026
 *  @brief SIMD-accelerated MaxSim, ColBERT late-interaction, for Sapphire Rapids AMX.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  bf16: fused AMX approach using TDPBF16PS for direct bf16 dot products, with per-tile column
 *  extraction for a running max of dot / ‖d‖ and angular distance finalization. Uses 4 accumulator
 *  tiles, TMM4-7, for 4-way document tile pipelining.
 *
 *  f32/f16: coarse i8 screening via AMX TDPBSSD, signed i8 × signed i8 → i32, with a 4-accumulator
 *  pipeline, weighted by each document's scale / ‖d‖, then full-precision refinement with
 *  nk_dot_f32/nk_dot_f16 of every document the screen's error bound cannot rule out.
 *
 *  TMM register allocation, all 3 dtypes:
 *  - TMM0: query, A-side — loaded once per depth step
 *  - TMM1: document, B-side — reloaded 4× per depth step, one per doc tile
 *  - TMM4: accumulator 0, doc tile 0
 *  - TMM5: accumulator 1, doc tile 1
 *  - TMM6: accumulator 2, doc tile 2
 *  - TMM7: accumulator 3, doc tile 3
 *  - TMM2, TMM3: unused
 *
 *  BF16 packed layout:
 *  [Header 64B] [0-63B padding for 64B alignment] [A-side tiles: col_tiles × depth_tiles × 1KB]
 *  [B-side tiles: col_tiles × depth_tiles × 1KB] [inverse norms: n × f32]
 *
 *  i8 packed layout, f32/f16:
 *  [Header 64B] [0-63B padding for 64B alignment] [i8 A-side tiles: col_tiles × depth_tiles × 1KB]
 *  [i8 B-side tiles: col_tiles × depth_tiles × 1KB] [originals 64B-aligned: n × original_stride]
 *  [inverse norms: n × f64] [screening weights: n × f32]
 *
 *  @verbatim
 *  Intrinsic                   Instruction         Notes
 *  _tile_dpbf16ps              TDPBF16PS           C += A × B (bf16 → f32), 16×16×32 MACs
 *  _tile_dpbssd                TDPBSSD             C += A × B (i8 × i8 → i32), 16×16×64 MACs
 *  _tile_loadd                 TILELOADD           Load tile from memory
 *  _tile_stored                TILESTORED          Store tile to memory
 *  _tile_zero                  TILEZERO            Zero a tile register
 *  @endverbatim
 */
#ifndef NUMKONG_MAXSIM_SAPPHIREAMX_H
#define NUMKONG_MAXSIM_SAPPHIREAMX_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SAPPHIREAMX_

#include "numkong/types.h"
#include "numkong/dots/sapphireamx.h" // AMX tile types, configure, load, transpose
#include "numkong/dot/skylake.h"      // `nk_dot_f32_through_f64_skylake_`, `nk_dot_f16_through_f32_skylake_`
#include "numkong/maxsim/serial.h"    // `nk_maxsim_screen_error_`, `nk_maxsim_screen_candidates_`
#include "numkong/maxsim/icelake.h"   // `nk_maxsim_pack_vector_f32_icelake_`
#include "numkong/spatial/haswell.h"  // `nk_rsqrt_f32x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                                  \
    __attribute__((target(                                                                                                                     \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,avx512bf16,avx512fp16,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", "avx512bf16", \
                   "avx512fp16", "f16c", "fma", "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8")
#endif

#pragma region I8 Header

/** i8 packed buffer header for AMX coarse+refine MaxSim (64 bytes). Stores both A-side (row-major)
 *  and B-side (quad-interleaved) i8 tile formats, original f32/f16 vectors for full-precision
 *  refinement, and per-vector inverse norms and screening weights. */
typedef struct {

    /** Number of vector-tile groups, ⌈n / 16⌉. */
    nk_u32_t column_tile_count;

    /** Number of depth tiles, ⌈depth / 64⌉, as TDPBSSD processes 64 i8 per tile. */
    nk_u32_t depth_tile_count;

    /** Actual vector count. */
    nk_u32_t columns;

    /** Actual depth (dimensions per vector). */
    nk_u32_t depth;

    /** Byte offset from buffer start to 64B-aligned A-side tiles. */
    nk_u32_t a_side_offset;

    /** Byte offset from buffer start to i8 B-side tiles. */
    nk_u32_t b_side_offset;

    /** Byte offset from buffer start to original f32/f16 vectors. */
    nk_u32_t originals_offset;

    /** 64B-aligned stride for originals. */
    nk_u32_t original_stride;

    /** Byte offset from buffer start to f64 inverse norms. */
    nk_u32_t norms_offset;

    /** Byte offset from buffer start to f32 screening weights, absmax / (127 · ‖v‖). */
    nk_u32_t screen_weights_offset;

    /** Padding to 64 bytes. */
    nk_u32_t reserved[4];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_maxsim_sapphireamx_i8_header_t;

nk_static_assert_(sizeof(nk_maxsim_sapphireamx_i8_header_t) == 64, nk_maxsim_sapphireamx_i8_header_must_be_64_bytes);

#pragma endregion I8 Header

#pragma region F32 Floats

/** Bytes of an F32 pack of @p vector_count vectors of @p depth: header, tiles, originals, norms. */
NUMKONG_INLINE nk_size_t nk_maxsim_packed_bytes_f32_sapphireamx_(nk_size_t vector_count, nk_size_t depth) {
    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 64);
    nk_size_t a_side_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t b_side_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f32_t), 64);
    nk_size_t originals_bytes = vector_count * original_stride;
    nk_size_t norms_bytes = vector_count * sizeof(nk_f64_t);
    nk_size_t screen_weights_bytes = vector_count * sizeof(nk_f32_t);
    return 64 + 63 + a_side_bytes + b_side_bytes + originals_bytes + norms_bytes + screen_weights_bytes;
}

#pragma endregion F32 Floats

#pragma region F16 Floats

/** Bytes of an F16 pack of @p vector_count vectors of @p depth: header, tiles, originals, norms. */
NUMKONG_INLINE nk_size_t nk_maxsim_packed_bytes_f16_sapphireamx_(nk_size_t vector_count, nk_size_t depth) {
    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 64);
    nk_size_t a_side_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t b_side_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f16_t), 64);
    nk_size_t originals_bytes = vector_count * original_stride;
    nk_size_t norms_bytes = vector_count * sizeof(nk_f64_t);
    nk_size_t screen_weights_bytes = vector_count * sizeof(nk_f32_t);
    return 64 + 63 + a_side_bytes + b_side_bytes + originals_bytes + norms_bytes + screen_weights_bytes;
}

#pragma endregion F16 Floats

#pragma region BF16 Floats

/** BF16 packed buffer header for AMX fused MaxSim (64 bytes). Stores both A-side (row-major) and
 *  B-side (pair-interleaved) tile formats, plus per-vector inverse norms for finalizing the angular
 *  distance of every query-document pair. */
typedef struct {

    /** Number of row-tile groups, ⌈n / 16⌉. */
    nk_u32_t column_tile_count;

    /** Number of depth tiles, ⌈depth / 32⌉, the depth granularity of BF16 TDPBF16PS. */
    nk_u32_t depth_tile_count;

    /** Actual vector count. */
    nk_u32_t columns;

    /** Actual depth (dimensions per vector). */
    nk_u32_t depth;

    /** Byte offset from buffer start to 64B-aligned A-side tiles. */
    nk_u32_t a_side_offset;

    /** Byte offset from buffer start to B-side tiles. */
    nk_u32_t b_side_offset;

    /** Byte offset from buffer start to inverse norms (f32). */
    nk_u32_t norms_offset;

    /** Padding to 64 bytes. */
    nk_u32_t reserved[7];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_maxsim_sapphireamx_bf16_header_t;

nk_static_assert_(sizeof(nk_maxsim_sapphireamx_bf16_header_t) == 64,
                  nk_maxsim_sapphireamx_bf16_header_must_be_64_bytes);

/** Bytes of a BF16 pack of @p vector_count vectors of @p depth: header, AMX tiles, norms. */
NUMKONG_INLINE nk_size_t nk_maxsim_packed_bytes_bf16_sapphireamx_(nk_size_t vector_count, nk_size_t depth) {
    nk_size_t const tile_bytes = 1024; // 16 × 32 × 2B = 1KB per tile
    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 32);
    nk_size_t a_side_bytes = column_tile_count * depth_tile_count * tile_bytes;
    nk_size_t b_side_bytes = column_tile_count * depth_tile_count * tile_bytes;
    nk_size_t norms_bytes = vector_count * sizeof(nk_f32_t);
    return sizeof(nk_maxsim_sapphireamx_bf16_header_t) + 63 + a_side_bytes + b_side_bytes + norms_bytes;
}

#pragma endregion BF16 Floats

/** Zeroes the @p bytes of a pack at @p packed with AVX-512 masked stores. */
NUMKONG_INLINE void nk_maxsim_zero_sapphireamx_(void *packed, nk_size_t bytes) {
    for (nk_size_t offset = 0; offset < bytes; offset += 64) {
        nk_size_t const chunk = bytes - offset < 64 ? bytes - offset : 64;
        _mm512_mask_storeu_epi8((char *)packed + offset, (__mmask64)_bzhi_u64(~(nk_u64_t)0, (unsigned int)chunk),
                                _mm512_setzero_si512());
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_sapphireamx(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_packed_bytes_f32_sapphireamx_(vector_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_sapphireamx(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sapphireamx_i8_header_t const *header = (nk_maxsim_sapphireamx_i8_header_t const *)packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_sapphireamx( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_zero_sapphireamx_(packed, nk_maxsim_packed_bytes_f32_sapphireamx_(vector_count, depth));

    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 64);
    nk_size_t original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f32_t), 64);
    nk_size_t a_side_total_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t b_side_total_bytes = column_tile_count * depth_tile_count * 1024;

    // Set up header — compute 64B-aligned A-side offset
    nk_maxsim_sapphireamx_i8_header_t *header = (nk_maxsim_sapphireamx_i8_header_t *)packed;
    nk_u32_t a_side_offset = (nk_u32_t)(nk_size_round_up_to_multiple_((nk_size_t)((char *)packed + 64), 64) -
                                        (nk_size_t)(char *)packed);
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)vector_count;
    header->depth = (nk_u32_t)depth;
    header->a_side_offset = a_side_offset;
    header->b_side_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes);
    header->originals_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes + b_side_total_bytes);
    header->original_stride = (nk_u32_t)original_stride;
    header->norms_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes + b_side_total_bytes +
                                      vector_count * original_stride);
    header->screen_weights_offset = (nk_u32_t)(header->norms_offset + vector_count * sizeof(nk_f64_t));
    header->capability = nk_cap_sapphireamx_k;
    for (nk_size_t reserved_index = 0; reserved_index < 4; reserved_index++) header->reserved[reserved_index] = 0;

    // Pointers to data regions (A-side is guaranteed 64B-aligned)
    nk_i8_t *a_side_base = (nk_i8_t *)((char *)packed + a_side_offset);
    char *b_side_base = (char *)packed + header->b_side_offset;
    char *originals_base = (char *)packed + header->originals_offset;
    nk_f64_t *inverse_norms = (nk_f64_t *)((char *)packed + header->norms_offset);
    nk_f32_t *screen_weights = (nk_f32_t *)((char *)packed + header->screen_weights_offset);

    // Copy originals and quantize into A-side tile rows, whose 64-code blocks sit one tile apart
    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        nk_maxsim_vector_metadata_t vector_metadata;
        nk_maxsim_pack_vector_f32_icelake_(
            (nk_f32_t const *)((char const *)vectors + vector_index * stride), depth,
            (nk_f32_t *)(originals_base + vector_index * original_stride),
            a_side_base + (vector_index / 16) * depth_tile_count * 1024 + (vector_index % 16) * 64, 1024,
            &vector_metadata);
        inverse_norms[vector_index] = vector_metadata.inverse_norm_f64;
        screen_weights[vector_index] = vector_metadata.screen_weight_f32;
    }

    // Transpose each A-side tile to B-side (both are 64B-aligned via header padding)
    for (nk_size_t tile_flat_index = 0; tile_flat_index < column_tile_count * depth_tile_count; tile_flat_index++) {
        nk_dots_i8_a16x64_sapphireamx_t const *a_tile =
            (nk_dots_i8_a16x64_sapphireamx_t const *)(a_side_base + tile_flat_index * 1024);
        nk_dots_i8_b64x16_sapphireamx_t *b_tile = (nk_dots_i8_b64x16_sapphireamx_t *)(b_side_base +
                                                                                      tile_flat_index * 1024);
        nk_dots_pack_i8_transposed_sapphireamx_(a_tile, b_tile);
    }
    return nk_success_k;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_sapphireamx_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_through_f64_skylake_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_sapphireamx_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_through_f32_skylake_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

/** Σ minⱼ angular(qᵢ, dⱼ) over two i8 packs: screens 16 × 64 blocks with TDPBSSD, then refines with
 *  @p refine_dot every document the screen cannot rule out, matching an exhaustive search. */
NUMKONG_INLINE nk_f64_t nk_maxsim_packed_i8_sapphireamx_(                                                   //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count, //
    nk_size_t depth, nk_maxsim_refine_dot_t refine_dot) {

    nk_maxsim_sapphireamx_i8_header_t const *query_header = (nk_maxsim_sapphireamx_i8_header_t const *)query_packed;
    nk_maxsim_sapphireamx_i8_header_t const *document_header =
        (nk_maxsim_sapphireamx_i8_header_t const *)document_packed;

    nk_size_t const depth_tile_count = query_header->depth_tile_count;
    nk_size_t const query_tile_count = query_header->column_tile_count;
    nk_size_t const document_tile_count = document_header->column_tile_count;

    // Query loads from A-side (64B-aligned), documents from B-side
    char const *query_a_side_base = (char const *)query_packed + query_header->a_side_offset;
    char const *document_b_side_base = (char const *)document_packed + document_header->b_side_offset;

    // Original vectors for refinement
    char const *query_originals = (char const *)query_packed + query_header->originals_offset;
    char const *document_originals = (char const *)document_packed + document_header->originals_offset;
    nk_size_t const query_original_stride = query_header->original_stride;
    nk_size_t const document_original_stride = document_header->original_stride;

    nk_f64_t const *query_inverse_norms = (nk_f64_t const *)((char const *)query_packed + query_header->norms_offset);
    nk_f64_t const *document_inverse_norms = (nk_f64_t const *)((char const *)document_packed +
                                                                document_header->norms_offset);
    nk_f32_t const *query_screen_weights = (nk_f32_t const *)((char const *)query_packed +
                                                              query_header->screen_weights_offset);
    nk_f32_t const *document_screen_weights = (nk_f32_t const *)((char const *)document_packed +
                                                                 document_header->screen_weights_offset);
    nk_f32_t const residue = 0.5f * nk_f32_sqrt_((nk_f32_t)depth);

    nk_amx_tile_configure_sapphireamx_();

    nk_f64_t total_angular_distance_f64 = 0.0, total_compensation_f64 = 0.0;

    for (nk_size_t query_tile_index = 0; query_tile_index < query_tile_count; query_tile_index++) {
        nk_size_t query_row_start = query_tile_index * 16;
        nk_size_t valid_queries = (query_row_start + 16 <= query_count) ? 16 : (query_count - query_row_start);

        nk_maxsim_screen_error_t errors[16];
        nk_f32_t lower_bounds[16];
        nk_f64_t best_cosines[16];
        for (nk_size_t query_in_tile = 0; query_in_tile < valid_queries; query_in_tile++) {
            errors[query_in_tile] = nk_maxsim_screen_error_(query_screen_weights[query_row_start + query_in_tile],
                                                            residue, depth);
            lower_bounds[query_in_tile] = NUMKONG_F32_MIN;
            best_cosines[query_in_tile] = NUMKONG_F32_MIN;
        }

        nk_align_(64) nk_i32_t group_dots[16][64];
        nk_u32_t candidates[64];
        nk_size_t document_tile_index = 0;

        // 4 document tiles at a time, then 1 at a time
        while (document_tile_index < document_tile_count) {
            nk_size_t const group_tiles = document_tile_index + 4 <= document_tile_count ? 4 : 1;
            if (group_tiles == 4) {
                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);

                for (nk_size_t depth_step_index = 0; depth_step_index < depth_tile_count; depth_step_index++) {
                    nk_size_t query_tile_flat_index = query_tile_index * depth_tile_count + depth_step_index;

                    _tile_loadd(0, (void const *)(query_a_side_base + query_tile_flat_index * 1024), 64);

                    nk_size_t document_tile_flat_0 = (document_tile_index + 0) * depth_tile_count + depth_step_index;
                    nk_size_t document_tile_flat_1 = (document_tile_index + 1) * depth_tile_count + depth_step_index;
                    nk_size_t document_tile_flat_2 = (document_tile_index + 2) * depth_tile_count + depth_step_index;
                    nk_size_t document_tile_flat_3 = (document_tile_index + 3) * depth_tile_count + depth_step_index;

                    _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_0 * 1024), 64);
                    _tile_dpbssd(4, 0, 1);
                    _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_1 * 1024), 64);
                    _tile_dpbssd(5, 0, 1);
                    _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_2 * 1024), 64);
                    _tile_dpbssd(6, 0, 1);
                    _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_3 * 1024), 64);
                    _tile_dpbssd(7, 0, 1);
                }

                // Row stride of a whole group, so each query's 64 dots land contiguous
                _tile_stored(4, &group_dots[0][0], 256);
                _tile_stored(5, &group_dots[0][16], 256);
                _tile_stored(6, &group_dots[0][32], 256);
                _tile_stored(7, &group_dots[0][48], 256);
            }
            else {
                _tile_zero(4);
                for (nk_size_t depth_step_index = 0; depth_step_index < depth_tile_count; depth_step_index++) {
                    nk_size_t query_tile_flat_index = query_tile_index * depth_tile_count + depth_step_index;
                    nk_size_t document_tile_flat_index = document_tile_index * depth_tile_count + depth_step_index;
                    _tile_loadd(0, (void const *)(query_a_side_base + query_tile_flat_index * 1024), 64);
                    _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_index * 1024), 64);
                    _tile_dpbssd(4, 0, 1);
                }
                _tile_stored(4, &group_dots[0][0], 256);
            }

            // Raise each query's lower bound over the group, then refine what it cannot rule out
            nk_size_t const group_start = document_tile_index * 16;
            nk_size_t const group_documents = group_start + group_tiles * 16 <= document_count
                                                  ? group_tiles * 16
                                                  : document_count - group_start;
            nk_f32_t const *weights = document_screen_weights + group_start;
            for (nk_size_t query_in_tile = 0; query_in_tile < valid_queries; query_in_tile++) {
                nk_size_t const query_index = query_row_start + query_in_tile;
                nk_maxsim_screen_lower_bound_(group_dots[query_in_tile], 1, weights, 1, group_documents,
                                              errors[query_in_tile], &lower_bounds[query_in_tile]);
                nk_size_t const candidate_count = nk_maxsim_screen_candidates_(group_dots[query_in_tile], 1, weights, 1,
                                                                               group_documents, errors[query_in_tile],
                                                                               lower_bounds[query_in_tile], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = group_start + candidates[candidate_index];
                    nk_f64_t const cosine = refine_dot(query_originals + query_index * query_original_stride,
                                                       document_originals + document_index * document_original_stride,
                                                       depth) *
                                            query_inverse_norms[query_index] * document_inverse_norms[document_index];
                    if (cosine > best_cosines[query_in_tile]) best_cosines[query_in_tile] = cosine;
                }
            }
            document_tile_index += group_tiles;
        }

        for (nk_size_t query_in_tile = 0; query_in_tile < valid_queries; query_in_tile++) {
            nk_f64_t angular = 1.0 - best_cosines[query_in_tile];
            if (angular < 0.0) angular = 0.0;
            nk_f64_dot2_(&total_angular_distance_f64, &total_compensation_f64, angular, 1.0);
        }
    }

    return total_angular_distance_f64 + total_compensation_f64;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_sapphireamx( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sapphireamx_i8_header_t const *)query_packed)->capability != nk_cap_sapphireamx_k ||
        ((nk_maxsim_sapphireamx_i8_header_t const *)document_packed)->capability != nk_cap_sapphireamx_k)
        return nk_pack_mismatch_k;
    *result = nk_maxsim_packed_i8_sapphireamx_(query_packed, document_packed, query_count, document_count, depth,
                                               nk_maxsim_refine_f32_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_sapphireamx(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_packed_bytes_f16_sapphireamx_(vector_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_sapphireamx(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sapphireamx_i8_header_t const *header = (nk_maxsim_sapphireamx_i8_header_t const *)packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_sapphireamx( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_zero_sapphireamx_(packed, nk_maxsim_packed_bytes_f16_sapphireamx_(vector_count, depth));

    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 64);
    nk_size_t original_stride = nk_size_round_up_to_multiple_(depth * sizeof(nk_f16_t), 64);
    nk_size_t a_side_total_bytes = column_tile_count * depth_tile_count * 1024;
    nk_size_t b_side_total_bytes = column_tile_count * depth_tile_count * 1024;

    // Set up header — compute 64B-aligned A-side offset
    nk_maxsim_sapphireamx_i8_header_t *header = (nk_maxsim_sapphireamx_i8_header_t *)packed;
    nk_u32_t a_side_offset = (nk_u32_t)(nk_size_round_up_to_multiple_((nk_size_t)((char *)packed + 64), 64) -
                                        (nk_size_t)(char *)packed);
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)vector_count;
    header->depth = (nk_u32_t)depth;
    header->a_side_offset = a_side_offset;
    header->b_side_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes);
    header->originals_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes + b_side_total_bytes);
    header->original_stride = (nk_u32_t)original_stride;
    header->norms_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes + b_side_total_bytes +
                                      vector_count * original_stride);
    header->screen_weights_offset = (nk_u32_t)(header->norms_offset + vector_count * sizeof(nk_f64_t));
    header->capability = nk_cap_sapphireamx_k;
    for (nk_size_t reserved_index = 0; reserved_index < 4; reserved_index++) header->reserved[reserved_index] = 0;

    // Pointers to data regions (A-side is guaranteed 64B-aligned)
    nk_i8_t *a_side_base = (nk_i8_t *)((char *)packed + a_side_offset);
    char *b_side_base = (char *)packed + header->b_side_offset;
    char *originals_base = (char *)packed + header->originals_offset;
    nk_f64_t *inverse_norms = (nk_f64_t *)((char *)packed + header->norms_offset);
    nk_f32_t *screen_weights = (nk_f32_t *)((char *)packed + header->screen_weights_offset);

    // Copy originals and quantize into A-side tile rows, whose 64-code blocks sit one tile apart
    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        nk_maxsim_vector_metadata_t vector_metadata;
        nk_maxsim_pack_vector_f16_icelake_(
            (nk_f16_t const *)((char const *)vectors + vector_index * stride), depth,
            (nk_f16_t *)(originals_base + vector_index * original_stride),
            a_side_base + (vector_index / 16) * depth_tile_count * 1024 + (vector_index % 16) * 64, 1024,
            &vector_metadata);
        inverse_norms[vector_index] = vector_metadata.inverse_norm_f64;
        screen_weights[vector_index] = vector_metadata.screen_weight_f32;
    }

    // Transpose each A-side tile to B-side (both are 64B-aligned via header padding)
    for (nk_size_t tile_flat_index = 0; tile_flat_index < column_tile_count * depth_tile_count; tile_flat_index++) {
        nk_dots_i8_a16x64_sapphireamx_t const *a_tile =
            (nk_dots_i8_a16x64_sapphireamx_t const *)(a_side_base + tile_flat_index * 1024);
        nk_dots_i8_b64x16_sapphireamx_t *b_tile = (nk_dots_i8_b64x16_sapphireamx_t *)(b_side_base +
                                                                                      tile_flat_index * 1024);
        nk_dots_pack_i8_transposed_sapphireamx_(a_tile, b_tile);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_sapphireamx( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sapphireamx_i8_header_t const *)query_packed)->capability != nk_cap_sapphireamx_k ||
        ((nk_maxsim_sapphireamx_i8_header_t const *)document_packed)->capability != nk_cap_sapphireamx_k)
        return nk_pack_mismatch_k;
    *result = (nk_f32_t)nk_maxsim_packed_i8_sapphireamx_(query_packed, document_packed, query_count, document_count,
                                                         depth, nk_maxsim_refine_f16_sapphireamx_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_sapphireamx(nk_size_t vector_count, nk_size_t depth,
                                                             nk_size_t *bytes) {
    *bytes = nk_maxsim_packed_bytes_bf16_sapphireamx_(vector_count, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_sapphireamx(void const *packed, nk_size_t *vectors,
                                                                nk_size_t *depth, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_sapphireamx_bf16_header_t const *header = (nk_maxsim_sapphireamx_bf16_header_t const *)packed;
    if (header->capability != nk_cap_sapphireamx_k) return nk_pack_mismatch_k;
    *vectors = header->columns;
    *depth = header->depth;
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_sapphireamx( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_zero_sapphireamx_(packed, nk_maxsim_packed_bytes_bf16_sapphireamx_(vector_count, depth));

    nk_size_t const tile_bytes = 1024;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t column_tile_count = nk_size_divide_round_up_(vector_count, 16);
    nk_size_t depth_tile_count = nk_size_divide_round_up_(depth, 32);

    // Set up header — compute 64B-aligned A-side offset
    nk_maxsim_sapphireamx_bf16_header_t *header = (nk_maxsim_sapphireamx_bf16_header_t *)packed;
    nk_u32_t a_side_offset = (nk_u32_t)(nk_size_round_up_to_multiple_(
                                            (nk_size_t)((char *)packed + sizeof(nk_maxsim_sapphireamx_bf16_header_t)),
                                            64) -
                                        (nk_size_t)(char *)packed);
    header->column_tile_count = (nk_u32_t)column_tile_count;
    header->depth_tile_count = (nk_u32_t)depth_tile_count;
    header->columns = (nk_u32_t)vector_count;
    header->depth = (nk_u32_t)depth;
    header->a_side_offset = a_side_offset;

    nk_size_t a_side_total_bytes = column_tile_count * depth_tile_count * tile_bytes;
    nk_size_t b_side_total_bytes = column_tile_count * depth_tile_count * tile_bytes;
    header->b_side_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes);
    header->norms_offset = (nk_u32_t)(a_side_offset + a_side_total_bytes + b_side_total_bytes);
    header->capability = nk_cap_sapphireamx_k;
    for (nk_size_t reserved_index = 0; reserved_index < 7; reserved_index++) header->reserved[reserved_index] = 0;

    // Pointers to data regions (A-side is guaranteed 64B-aligned)
    char *a_side_base = (char *)packed + a_side_offset;
    char *b_side_base = (char *)packed + header->b_side_offset;
    nk_f32_t *inverse_norms = (nk_f32_t *)((char *)packed + header->norms_offset);

    // Pack tiles: for each column tile × depth tile, store both A-side and B-side
    for (nk_size_t column_tile_index = 0; column_tile_index < column_tile_count; column_tile_index++) {
        nk_size_t row_start = column_tile_index * 16;
        nk_size_t valid_rows = (row_start + 16 <= vector_count) ? 16 : (vector_count - row_start);

        for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
            nk_size_t depth_start = depth_tile_index * 32;
            nk_size_t valid_columns = (depth_start + 32 <= depth) ? 32 : (depth - depth_start);

            nk_size_t tile_flat_index = column_tile_index * depth_tile_count + depth_tile_index;

            // Load source vectors into A-side tile (row-major, zero-padded)
            nk_dots_bf16_a16x32_sapphireamx_t a_tile;
            nk_dots_bf16_load_a_sapphireamx_(&a_tile, vectors + row_start * stride_elements + depth_start,
                                             stride_elements, valid_rows, valid_columns);

            // Transpose to the pair-interleaved B-side tile, then store both 64-byte aligned tiles
            nk_dots_bf16_b32x16_sapphireamx_t b_tile;
            nk_dots_pack_bf16_transposed_sapphireamx_(&a_tile, &b_tile);
            for (nk_size_t row = 0; row < 16; row++) {
                _mm512_store_si512(a_side_base + tile_flat_index * tile_bytes + row * 64,
                                   _mm512_load_si512(a_tile.data[row]));
                _mm512_store_si512(b_side_base + tile_flat_index * tile_bytes + row * 64,
                                   _mm512_load_si512(b_tile.data[row]));
            }
        }
    }

    // Norms read each vector back from its A-side tile rows
    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *vector_tiles = a_side_base + (vector_index / 16) * depth_tile_count * tile_bytes +
                                   (vector_index % 16) * 64;
        __m512 norm_squared_f32x16 = _mm512_setzero_ps();
        for (nk_size_t depth_tile_index = 0; depth_tile_index < depth_tile_count; depth_tile_index++) {
            __m512i const values_bf16x32 = _mm512_load_si512(vector_tiles + depth_tile_index * tile_bytes);
            __m512 const low_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm512_castsi512_si256(values_bf16x32));
            __m512 const high_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm512_extracti64x4_epi64(values_bf16x32, 1));
            norm_squared_f32x16 = _mm512_fmadd_ps(low_f32x16, low_f32x16, norm_squared_f32x16);
            norm_squared_f32x16 = _mm512_fmadd_ps(high_f32x16, high_f32x16, norm_squared_f32x16);
        }
        nk_f32_t const norm_squared_f32 = nk_reduce_add_f32x16_skylake_(norm_squared_f32x16);
        inverse_norms[vector_index] = (norm_squared_f32 > 0.0f)
                                          ? _mm_cvtss_f32(nk_rsqrt_f32x4_haswell_(_mm_set_ss(norm_squared_f32)))
                                          : 0.0f;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_sapphireamx( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_maxsim_sapphireamx_bf16_header_t const *)query_packed)->capability != nk_cap_sapphireamx_k ||
        ((nk_maxsim_sapphireamx_bf16_header_t const *)document_packed)->capability != nk_cap_sapphireamx_k)
        return nk_pack_mismatch_k;

    nk_unused_(depth); // tile counts from header encode depth

    nk_maxsim_sapphireamx_bf16_header_t const *query_header = (nk_maxsim_sapphireamx_bf16_header_t const *)query_packed;
    nk_maxsim_sapphireamx_bf16_header_t const *document_header =
        (nk_maxsim_sapphireamx_bf16_header_t const *)document_packed;

    nk_size_t const depth_tile_count = query_header->depth_tile_count;
    nk_size_t const query_column_tile_count = query_header->column_tile_count;
    nk_size_t const document_column_tile_count = document_header->column_tile_count;

    // Query loads from A-side tiles (64B-aligned), documents from B-side tiles
    char const *query_a_side_base = (char const *)query_packed + query_header->a_side_offset;
    char const *document_b_side_base = (char const *)document_packed + document_header->b_side_offset;

    nk_f32_t const *query_inverse_norms = (nk_f32_t const *)((char const *)query_packed + query_header->norms_offset);
    nk_f32_t const *document_inverse_norms = (nk_f32_t const *)((char const *)document_packed +
                                                                document_header->norms_offset);

    nk_amx_tile_configure_sapphireamx_();

    // Gather indices for column extraction from 16×16 f32 tile:
    // tile_result[row][col] is at f32 offset row*16 + col
    __m512i const row_stride_indices_i32x16 = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192,
                                                                208, 224, 240);

    nk_f64_t total_angular_distance_f64 = 0.0;

    for (nk_size_t query_tile_index = 0; query_tile_index < query_column_tile_count; query_tile_index++) {
        nk_size_t query_row_start = query_tile_index * 16;
        nk_size_t valid_queries = (query_row_start + 16 <= query_count) ? 16 : (query_count - query_row_start);
        __mmask16 valid_query_m16 = (valid_queries >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << valid_queries) - 1);

        __m512 running_maximum_f32x16 = _mm512_set1_ps(NUMKONG_F32_MIN);

        nk_align_(64) nk_f32_t tile_results_f32[4][16][16];
        nk_size_t document_tile_index = 0;

        // Fast path: 4 document tiles at a time using TMM4-7
        for (; document_tile_index + 4 <= document_column_tile_count; document_tile_index += 4) {
            _tile_zero(4);
            _tile_zero(5);
            _tile_zero(6);
            _tile_zero(7);

            for (nk_size_t depth_step_index = 0; depth_step_index < depth_tile_count; depth_step_index++) {
                nk_size_t query_tile_flat_index = query_tile_index * depth_tile_count + depth_step_index;

                _tile_loadd(0, (void const *)(query_a_side_base + query_tile_flat_index * 1024), 64);

                nk_size_t document_tile_flat_0 = (document_tile_index + 0) * depth_tile_count + depth_step_index;
                nk_size_t document_tile_flat_1 = (document_tile_index + 1) * depth_tile_count + depth_step_index;
                nk_size_t document_tile_flat_2 = (document_tile_index + 2) * depth_tile_count + depth_step_index;
                nk_size_t document_tile_flat_3 = (document_tile_index + 3) * depth_tile_count + depth_step_index;

                _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_0 * 1024), 64);
                _tile_dpbf16ps(4, 0, 1);
                _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_1 * 1024), 64);
                _tile_dpbf16ps(5, 0, 1);
                _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_2 * 1024), 64);
                _tile_dpbf16ps(6, 0, 1);
                _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_3 * 1024), 64);
                _tile_dpbf16ps(7, 0, 1);
            }

            _tile_stored(4, tile_results_f32[0], 64);
            _tile_stored(5, tile_results_f32[1], 64);
            _tile_stored(6, tile_results_f32[2], 64);
            _tile_stored(7, tile_results_f32[3], 64);

            // Column extraction from 4 tiles
            for (nk_size_t tile_offset = 0; tile_offset < 4; tile_offset++) {
                nk_size_t document_column_start = (document_tile_index + tile_offset) * 16;
                nk_size_t valid_documents = (document_column_start + 16 <= document_count)
                                                ? 16
                                                : (document_count - document_column_start);
                for (nk_size_t column_within_tile = 0; column_within_tile < valid_documents; column_within_tile++) {
                    __m512i gather_index_i32x16 = _mm512_add_epi32(row_stride_indices_i32x16,
                                                                   _mm512_set1_epi32((int)column_within_tile));
                    __m512 column_dots_f32x16 = _mm512_i32gather_ps(gather_index_i32x16,
                                                                    (float const *)tile_results_f32[tile_offset], 4);
                    __m512 column_scores_f32x16 = _mm512_mul_ps(
                        column_dots_f32x16,
                        _mm512_set1_ps(document_inverse_norms[document_column_start + column_within_tile]));
                    running_maximum_f32x16 = _mm512_max_ps(running_maximum_f32x16, column_scores_f32x16);
                }
            }
        }

        // Remainder: 1 document tile at a time using TMM4 only
        for (; document_tile_index < document_column_tile_count; document_tile_index++) {
            nk_size_t document_column_start = document_tile_index * 16;
            nk_size_t valid_documents = (document_column_start + 16 <= document_count)
                                            ? 16
                                            : (document_count - document_column_start);

            _tile_zero(4);

            for (nk_size_t depth_step_index = 0; depth_step_index < depth_tile_count; depth_step_index++) {
                nk_size_t query_tile_flat_index = query_tile_index * depth_tile_count + depth_step_index;
                nk_size_t document_tile_flat_index = document_tile_index * depth_tile_count + depth_step_index;

                _tile_loadd(0, (void const *)(query_a_side_base + query_tile_flat_index * 1024), 64);
                _tile_loadd(1, (void const *)(document_b_side_base + document_tile_flat_index * 1024), 64);
                _tile_dpbf16ps(4, 0, 1);
            }

            _tile_stored(4, tile_results_f32[0], 64);

            for (nk_size_t column_within_tile = 0; column_within_tile < valid_documents; column_within_tile++) {
                __m512i gather_index_i32x16 = _mm512_add_epi32(row_stride_indices_i32x16,
                                                               _mm512_set1_epi32((int)column_within_tile));
                __m512 column_dots_f32x16 = _mm512_i32gather_ps(gather_index_i32x16, (float const *)tile_results_f32[0],
                                                                4);
                __m512 column_scores_f32x16 = _mm512_mul_ps(
                    column_dots_f32x16,
                    _mm512_set1_ps(document_inverse_norms[document_column_start + column_within_tile]));
                running_maximum_f32x16 = _mm512_max_ps(running_maximum_f32x16, column_scores_f32x16);
            }
        }

        // Angular distance finalization using AVX-512
        __m512 query_inverse_norms_f32x16 = _mm512_maskz_loadu_ps(valid_query_m16,
                                                                  query_inverse_norms + query_row_start);

        // cosine = max(dot × inv_norm_d) × inv_norm_q
        __m512 cosine_f32x16 = _mm512_mul_ps(running_maximum_f32x16, query_inverse_norms_f32x16);

        // angular = max(1 - cosine, 0), masked to valid queries only
        __m512 angular_distance_f32x16 = _mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), cosine_f32x16),
                                                       _mm512_setzero_ps());
        angular_distance_f32x16 = _mm512_maskz_mov_ps(valid_query_m16, angular_distance_f32x16);

        total_angular_distance_f64 += (nk_f64_t)_mm512_reduce_add_ps(angular_distance_f32x16);
    }

    *result = (nk_f32_t)total_angular_distance_f64;
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SAPPHIREAMX_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MAXSIM_SAPPHIREAMX_H
