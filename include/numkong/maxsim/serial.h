/**
 *  @file include/numkong/maxsim/serial.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *  @brief SWAR-accelerated MaxSim, ColBERT late-interaction, for SIMD-free CPUs.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Defines the packed buffer header and per-vector metadata structures used by all MaxSim ISA
 *  backends, plus scalar reference implementations for correctness validation.
 *
 *  MaxSim computes: result = Σᵢ minⱼ angular(qᵢ, dⱼ) — angular distance late-interaction scoring.
 *
 *  Strategy: coarse i8-quantized screening of i8 dots, each weighted by its document's scale / ‖d‖
 *  to rank by cosine, then full-precision refinement via existing nk_dot_* primitives of every
 *  document the screen cannot rule out, finalized with angular distance:
 *  1 - dot / sqrt(||q||² × ||d||²).
 *
 *  @section packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors 64B-aligned] [metadata 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  - i8 region: row-major with padded depth for SIMD alignment
 *  - Metadata region: vector_count x 16 bytes (screening weight + sum + inverse norm per vector)
 *  - Originals region: row-major bf16 or f32, stride padded to 64B for nk_dot_* calls
 */
#ifndef NUMKONG_MAXSIM_SERIAL_H
#define NUMKONG_MAXSIM_SERIAL_H

#include "numkong/types.h"
#include "numkong/capabilities.h"   // `nk_capability_t`
#include "numkong/cast/serial.h"    // `nk_bf16_to_f32_`
#include "numkong/dot/serial.h"     // `nk_dot_bf16_`, `nk_dot_f32_`, `nk_dot_f16_`
#include "numkong/spatial/serial.h" // `nk_f32_rsqrt_`

#if defined(__cplusplus)
extern "C" {
#endif

/** Packed buffer header, 64 bytes and cache-line aligned, opening every MaxSim packed buffer. */
typedef struct {

    /** Number of vectors packed. */
    nk_u32_t vectors;

    /** Logical depth (number of elements per vector). */
    nk_u32_t depth;

    /** Padded i8 depth in bytes (SIMD-aligned). */
    nk_u32_t depth_i8_padded;

    /** Size of each original element in bytes: 2 for bf16, 4 for f32. */
    nk_u32_t original_element_bytes;

    /** Byte offset from buffer start to i8 region. */
    nk_u32_t offset_i8_data;

    /** Byte offset from buffer start to metadata region. */
    nk_u32_t offset_metadata;

    /** Byte offset from buffer start to originals region. */
    nk_u32_t offset_original_data;

    /** Row stride in bytes for originals region. */
    nk_u32_t original_stride_bytes;

    /** Padding to 64 bytes. */
    nk_u32_t reserved[6];

    /** The capability that packed the buffer, which every consumer checks. */
    nk_capability_t capability;
} nk_maxsim_packed_header_t;

nk_static_assert_(sizeof(nk_maxsim_packed_header_t) == 64, nk_maxsim_packed_header_must_be_64_bytes);

/** Per-vector quantization metadata, 16 bytes, stored once per vector in the packed buffer's
 *  metadata region. */
typedef struct {

    /** Screening weight scale / ‖v‖ for scale absmax / range_limit, so an i8 dot times it ranks
     *  by cosine; 0 for a zero vector. */
    nk_f32_t screen_weight_f32;

    /** Sum of all i8 quantized elements (for VPDPBUSD/VPMADDUBSW bias correction). */
    nk_i32_t sum_i8_i32;

    /** Inverse norm 1 / ‖v‖ for angular scores, from an f64 sum of squares; 0 for a zero vector. */
    nk_f64_t inverse_norm_f64;
} nk_maxsim_vector_metadata_t;

nk_static_assert_(sizeof(nk_maxsim_vector_metadata_t) == 16, nk_maxsim_vector_metadata_must_be_16_bytes);

/** Conversion function pointer type for element-to-f32 conversion. Each conversion reads one
 *  element from @c source and writes one f32 to @c destination. */
typedef void (*nk_maxsim_to_f32_t)(void const *source, nk_f32_t *destination);

/** Identity conversion for f32 sources — just a typed memcpy. */
NUMKONG_INLINE void nk_f32_to_f32_(void const *source, nk_f32_t *destination) {
    *destination = *(nk_f32_t const *)source;
}

/** Fills the packed buffer header, recording the packing @p capability, and returns the padded
 *  i8 depth. Consolidates header/offset computation duplicated in every pack function. */
NUMKONG_INLINE nk_size_t nk_maxsim_packed_header_setup_(   //
    void *packed, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t depth_simd_dimensions, nk_size_t original_element_bytes, nk_capability_t capability) {

    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += depth_simd_dimensions;

    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(vector_count * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        vector_count * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * original_element_bytes, 64);

    // Zero the whole buffer up front so the alignment slack between the i8 / metadata / originals
    // regions and each region's round-up tail stay deterministic — the pack writes only the data,
    // leaving these bytes zero, which makes the packed blob a pure function of its inputs.

    nk_size_t const total_bytes = header_size + i8_region_size + metadata_region_size + vector_count * original_stride;
    for (nk_size_t byte_index = 0; byte_index < total_bytes; byte_index++) ((char *)packed)[byte_index] = 0;

    nk_maxsim_packed_header_t *header = (nk_maxsim_packed_header_t *)packed;
    header->vectors = (nk_u32_t)vector_count;
    header->depth = (nk_u32_t)depth;
    header->depth_i8_padded = (nk_u32_t)depth_i8_padded;
    header->original_element_bytes = (nk_u32_t)original_element_bytes;
    header->offset_i8_data = (nk_u32_t)header_size;
    header->offset_metadata = (nk_u32_t)(header_size + i8_region_size);
    header->offset_original_data = (nk_u32_t)(header_size + i8_region_size + metadata_region_size);
    header->original_stride_bytes = (nk_u32_t)original_stride;
    header->capability = capability;
    for (nk_size_t reserved_index = 0; reserved_index < 6; reserved_index++) header->reserved[reserved_index] = 0;

    return depth_i8_padded;
}

/** Quantizes a single source vector to i8 and computes its metadata. It calls the conversion
 *  callback element by element, so it needs no scratch buffer and works for any depth. */
NUMKONG_INLINE void nk_maxsim_quantize_vector_(                          //
    void const *source_vector, nk_size_t element_bytes, nk_size_t depth, //
    nk_size_t depth_i8_padded, nk_f32_t scale_limit,                     //
    nk_maxsim_to_f32_t convert_to_f32,                                   //
    nk_i8_t *destination_i8, nk_maxsim_vector_metadata_t *metadata) {

    char const *source_bytes = (char const *)source_vector;

    // Pass 1: Find absmax, compute norm_squared
    nk_f32_t absmax_f32 = 0.0f;
    nk_f64_t norm_squared_f64 = 0.0;
    for (nk_size_t dim_index = 0; dim_index < depth; dim_index++) {
        nk_f32_t value_f32;
        convert_to_f32(source_bytes + dim_index * element_bytes, &value_f32);
        nk_f32_t abs_value = nk_f32_abs_(value_f32);
        if (abs_value > absmax_f32) absmax_f32 = abs_value;
        norm_squared_f64 += (nk_f64_t)value_f32 * value_f32;
    }

    nk_f32_t scale_f32 = absmax_f32 / scale_limit;
    if (scale_f32 == 0.0f) scale_f32 = 1.0f;

    // Pass 2: Quantize to i8 and compute sum
    nk_i32_t sum_quantized_i32 = 0;
    for (nk_size_t dim_index = 0; dim_index < depth; dim_index++) {
        nk_f32_t value_f32;
        convert_to_f32(source_bytes + dim_index * element_bytes, &value_f32);
        nk_f32_t scaled = value_f32 / scale_f32;
        nk_i32_t quantized_value;
        if (scaled >= 0.0f) quantized_value = (nk_i32_t)(scaled + 0.5f);
        else quantized_value = (nk_i32_t)(scaled - 0.5f);
        if (quantized_value > (nk_i32_t)scale_limit) quantized_value = (nk_i32_t)scale_limit;
        if (quantized_value < -(nk_i32_t)scale_limit) quantized_value = -(nk_i32_t)scale_limit;

        destination_i8[dim_index] = (nk_i8_t)quantized_value;
        sum_quantized_i32 += quantized_value;
    }

    // Zero-pad remaining bytes
    for (nk_size_t dim_index = depth; dim_index < depth_i8_padded; dim_index++) destination_i8[dim_index] = 0;

    metadata->inverse_norm_f64 = norm_squared_f64 > 0.0 ? nk_f64_rsqrt_(norm_squared_f64) : 0.0;
    metadata->screen_weight_f32 = scale_f32 * (nk_f32_t)metadata->inverse_norm_f64;
    metadata->sum_i8_i32 = sum_quantized_i32;
}

/** Region pointers extracted from two packed buffers. Eliminates ~15 lines of boilerplate per
 *  compute function. */
typedef struct {
    nk_size_t depth_i8_padded;
    nk_i8_t const *query_quantized;
    nk_i8_t const *document_quantized;
    nk_maxsim_vector_metadata_t const *query_metadata;
    nk_maxsim_vector_metadata_t const *document_metadata;
    char const *query_originals;
    char const *document_originals;
    nk_size_t query_original_stride;
    nk_size_t document_original_stride;
} nk_maxsim_packed_regions_t;

NUMKONG_INLINE nk_maxsim_packed_regions_t nk_maxsim_extract_packed_regions_( //
    void const *query_packed, void const *document_packed) {

    nk_maxsim_packed_header_t const *query_header = (nk_maxsim_packed_header_t const *)query_packed;
    nk_maxsim_packed_header_t const *document_header = (nk_maxsim_packed_header_t const *)document_packed;

    nk_maxsim_packed_regions_t regions;
    regions.depth_i8_padded = query_header->depth_i8_padded;
    regions.query_quantized = (nk_i8_t const *)((char const *)query_packed + query_header->offset_i8_data);
    regions.document_quantized = (nk_i8_t const *)((char const *)document_packed + document_header->offset_i8_data);
    regions.query_metadata = (nk_maxsim_vector_metadata_t const *)((char const *)query_packed +
                                                                   query_header->offset_metadata);
    regions.document_metadata = (nk_maxsim_vector_metadata_t const *)((char const *)document_packed +
                                                                      document_header->offset_metadata);
    regions.query_originals = (char const *)query_packed + query_header->offset_original_data;
    regions.document_originals = (char const *)document_packed + document_header->offset_original_data;
    regions.query_original_stride = query_header->original_stride_bytes;
    regions.document_original_stride = document_header->original_stride_bytes;
    return regions;
}

/**
 *  @brief Computes padded i8 depth and total packed buffer size for maxsim.
 *
 *  The layout is the header, then the i8 data, the metadata and the originals, each 64B-aligned.
 *
 *  @param[in] vector_count Number of vectors to pack.
 *  @param[in] depth Number of elements per vector.
 *  @param[in] original_element_bytes Size of each original element (2 for bf16, 4 for f32).
 *  @param[in] depth_simd_dimensions SIMD width for i8 depth padding (1 for serial).
 */
NUMKONG_INLINE nk_size_t nk_maxsim_pack_size_( //
    nk_size_t vector_count, nk_size_t depth,   //
    nk_size_t original_element_bytes, nk_size_t depth_simd_dimensions) {

    // Pad i8 depth to SIMD width
    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);

    // Break power-of-2 strides for cache associativity
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += depth_simd_dimensions;

    // Calculate region sizes
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(vector_count * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        vector_count * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * original_element_bytes, 64);
    nk_size_t const originals_region_size = vector_count * original_stride;

    return header_size + i8_region_size + metadata_region_size + originals_region_size;
}

/**
 *  @brief Reads a packed maxsim buffer's shape from its header.
 *
 *  Shared by every per-(dtype, ISA) nk_maxsim_packed_shape_* accessor.
 */
NUMKONG_INLINE void nk_maxsim_packed_shape_(void const *packed, nk_size_t *vectors, nk_size_t *depth) {
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    *vectors = header->vectors;
    *depth = header->depth;
}

/** How far one query's screened score s = dot_i8 · w_d may stray from dot(q, d) / (scale_q · ‖d‖),
 *  as @c base + w_d · @c per_weight. */
typedef struct {
    nk_f32_t base;
    nk_f32_t per_weight;
} nk_maxsim_screen_error_t;

/** Rounding residues e_q, e_d of norm ≤ r = ½√depth give |error| ≤ r + w_d · (r / w_q + r²).
 *  It holds as ‖q / scale_q‖ = 1 / w_q and ‖d / scale_d‖ · w_d = 1.
 *  The extra depth · 2⁻²¹ / w_q term absorbs the f32 norm rounding. */
NUMKONG_INLINE nk_maxsim_screen_error_t nk_maxsim_screen_error_( //
    nk_f32_t query_screen_weight, nk_f32_t residue, nk_size_t depth) NUMKONG_STREAMABLE_ {
    nk_f32_t const query_scaled_norm = query_screen_weight > 0.0f ? 1.0f / query_screen_weight : 0.0f;
    nk_maxsim_screen_error_t error;
    error.base = residue + (nk_f32_t)depth / 2097152.0f * query_scaled_norm;
    error.per_weight = residue * (query_scaled_norm + residue);
    return error;
}

/** Raises @p lower_bound to the best screened score, less its error, among @p document_count dots
 *  spaced @p dots_stride apart, weighted by screening weights spaced @p weights_stride apart. */
NUMKONG_INLINE void nk_maxsim_screen_lower_bound_(                                                  //
    nk_i32_t const *dots, nk_size_t dots_stride, nk_f32_t const *weights, nk_size_t weights_stride, //
    nk_size_t document_count, nk_maxsim_screen_error_t error, nk_f32_t *lower_bound) NUMKONG_STREAMABLE_ {
    nk_f32_t bound = *lower_bound;
    for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
        nk_f32_t const weight = weights[document_index * weights_stride];
        nk_f32_t const low = (nk_f32_t)dots[document_index * dots_stride] * weight - error.base -
                             weight * error.per_weight;
        if (low > bound) bound = low;
    }
    *lower_bound = bound;
}

/** Lists into @p candidates the documents whose screened score plus error reaches @p lower_bound,
 *  the only ones that can hold the exact maximum, and returns their count. */
NUMKONG_INLINE nk_size_t nk_maxsim_screen_candidates_(                                              //
    nk_i32_t const *dots, nk_size_t dots_stride, nk_f32_t const *weights, nk_size_t weights_stride, //
    nk_size_t document_count, nk_maxsim_screen_error_t error, nk_f32_t lower_bound,                 //
    nk_u32_t *candidates) NUMKONG_STREAMABLE_ {
    nk_size_t candidate_count = 0;
    for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
        nk_f32_t const weight = weights[document_index * weights_stride];
        nk_f32_t const high = (nk_f32_t)dots[document_index * dots_stride] * weight + error.base +
                              weight * error.per_weight;
        if (high >= lower_bound) candidates[candidate_count++] = (nk_u32_t)document_index;
    }
    return candidate_count;
}

/** Coarse kernel writing the i8 dots of @p query_count queries against @p document_count documents
 *  into @p dots, query-major. */
typedef void (*nk_maxsim_coarse_dots_t)(nk_i8_t const *query_i8, nk_i8_t const *document_i8,
                                        nk_maxsim_vector_metadata_t const *document_metadata, nk_size_t query_count,
                                        nk_size_t document_count, nk_size_t depth_i8_padded, nk_i32_t *dots);

/** Full-precision dot product of two original vectors of @p depth elements. */
typedef nk_f64_t (*nk_maxsim_refine_dot_t)(void const *query, void const *document, nk_size_t depth);

/** Σ minⱼ angular(qᵢ, dⱼ) over two packs sharing @c nk_maxsim_packed_header_t: screens 32 × 128
 *  tiles with @p coarse_dots, then refines with @p refine_dot every document the screen cannot rule
 *  out, so the result matches an exhaustive search. */
NUMKONG_INLINE nk_f64_t nk_maxsim_packed_angular_(                                  //
    void const *query_packed, void const *document_packed, nk_size_t query_count,   //
    nk_size_t document_count, nk_size_t depth, nk_maxsim_coarse_dots_t coarse_dots, //
    nk_maxsim_refine_dot_t refine_dot) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_size_t const weights_stride = sizeof(nk_maxsim_vector_metadata_t) / sizeof(nk_f32_t);
    nk_f32_t const residue = 0.5f * nk_f32_sqrt_((nk_f32_t)depth);
    nk_i32_t dots[32 * 128];
    nk_u32_t candidates[128];
    nk_f64_t total_angular_distance = 0.0, total_compensation = 0.0;

    for (nk_size_t query_start = 0; query_start < query_count; query_start += 32) {
        nk_size_t const query_tile = query_count - query_start < 32 ? query_count - query_start : 32;
        nk_maxsim_screen_error_t errors[32];
        nk_f32_t lower_bounds[32];
        nk_f64_t best_cosines[32];
        for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
            errors[query_index] = nk_maxsim_screen_error_(
                regions.query_metadata[query_start + query_index].screen_weight_f32, residue, depth);
            lower_bounds[query_index] = NUMKONG_F32_MIN;
            best_cosines[query_index] = NUMKONG_F32_MIN;
        }

        for (nk_size_t document_start = 0; document_start < document_count; document_start += 128) {
            nk_size_t const document_tile = document_count - document_start < 128 ? document_count - document_start
                                                                                  : 128;
            coarse_dots(regions.query_quantized + query_start * regions.depth_i8_padded,
                        regions.document_quantized + document_start * regions.depth_i8_padded,
                        regions.document_metadata + document_start, query_tile, document_tile, regions.depth_i8_padded,
                        dots);
            nk_f32_t const *weights = &regions.document_metadata[document_start].screen_weight_f32;

            for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
                nk_size_t const query_global_index = query_start + query_index;
                nk_i32_t const *query_dots = dots + query_index * document_tile;
                nk_maxsim_screen_lower_bound_(query_dots, 1, weights, weights_stride, document_tile,
                                              errors[query_index], &lower_bounds[query_index]);
                nk_size_t const candidate_count = nk_maxsim_screen_candidates_(query_dots, 1, weights, weights_stride,
                                                                               document_tile, errors[query_index],
                                                                               lower_bounds[query_index], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = document_start + candidates[candidate_index];
                    nk_f64_t const cosine =
                        refine_dot(regions.query_originals + query_global_index * regions.query_original_stride,
                                   regions.document_originals + document_index * regions.document_original_stride,
                                   depth) *
                        regions.query_metadata[query_global_index].inverse_norm_f64 *
                        regions.document_metadata[document_index].inverse_norm_f64;
                    if (cosine > best_cosines[query_index]) best_cosines[query_index] = cosine;
                }
            }
        }

        for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
            nk_f64_t angular = 1.0 - best_cosines[query_index];
            if (angular < 0.0) angular = 0.0;
            nk_f64_dot2_(&total_angular_distance, &total_compensation, angular, 1.0);
        }
    }
    return total_angular_distance + total_compensation;
}

/** Whether @p packed was packed by @p capability, as every pack sharing this header records. */
NUMKONG_INLINE int nk_maxsim_packed_by_(void const *packed, nk_capability_t capability) {
    return ((nk_maxsim_packed_header_t const *)packed)->capability == capability;
}

/** Whether @p packed_shape reads back from @p packed the @p width and @p depth a packed kernel is
 *  about to trust. */
NUMKONG_CONSTEXPR int nk_packed_shape_matches_(void (*packed_shape)(void const *, nk_size_t *, nk_size_t *),
                                               void const *packed, nk_size_t width, nk_size_t depth) {
    nk_size_t packed_width = 0, packed_depth = 0;
    packed_shape(packed, &packed_width, &packed_depth);
    return packed_width == width && packed_depth == depth;
}

#if NUMKONG_TARGET_SERIAL

/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_bf16_t), 1);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f32_t), 1);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_serial( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes,
                                                               nk_cap_serial_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_serial( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes,
                                                               nk_cap_serial_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_serial(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f16_t), 1);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_serial(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_serial_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_serial( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes,
                                                               nk_cap_serial_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_SERIAL

/** DType-agnostic coarse i8 kernel for the serial backend, writing the signed i8 × i8 dot products,
 *  which need no bias correction here, as an @c nk_maxsim_coarse_dots_t. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_serial_(        //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {
    nk_unused_(document_metadata);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32[4] = {0, 0, 0, 0};

            for (nk_size_t dim_index = 0; dim_index < depth_i8_padded; dim_index++) {
                nk_i32_t document_value = (nk_i32_t)document_i8_row[dim_index];
                accumulator_i32[0] += (nk_i32_t)query_i8[(query_block_start_index + 0) * depth_i8_padded + dim_index] *
                                      document_value;
                accumulator_i32[1] += (nk_i32_t)query_i8[(query_block_start_index + 1) * depth_i8_padded + dim_index] *
                                      document_value;
                accumulator_i32[2] += (nk_i32_t)query_i8[(query_block_start_index + 2) * depth_i8_padded + dim_index] *
                                      document_value;
                accumulator_i32[3] += (nk_i32_t)query_i8[(query_block_start_index + 3) * depth_i8_padded + dim_index] *
                                      document_value;
            }

            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                dots[(query_block_start_index + query_tile_index) * document_count + document_index] =
                    accumulator_i32[query_tile_index];
        }
    }

    // Edge path: remaining 1-3 queries
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32 = 0;

            for (nk_size_t dim_index = 0; dim_index < depth_i8_padded; dim_index++)
                accumulator_i32 += (nk_i32_t)query_i8_row[dim_index] * (nk_i32_t)document_i8_row[dim_index];

            dots[query_index * document_count + document_index] = accumulator_i32;
        }
    }
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_serial_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_serial_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_serial_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

#if NUMKONG_TARGET_SERIAL

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_serial_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_serial_k))
        return nk_pack_mismatch_k;

    nk_assert_(nk_packed_shape_matches_(nk_maxsim_packed_shape_, query_packed, query_count, depth) &&
               nk_packed_shape_matches_(nk_maxsim_packed_shape_, document_packed, document_count, depth));
    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_serial_, nk_maxsim_refine_bf16_serial_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_serial_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_serial_k))
        return nk_pack_mismatch_k;

    nk_assert_(nk_packed_shape_matches_(nk_maxsim_packed_shape_, query_packed, query_count, depth) &&
               nk_packed_shape_matches_(nk_maxsim_packed_shape_, document_packed, document_count, depth));
    *result = nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                        nk_maxsim_coarse_dots_serial_, nk_maxsim_refine_f32_serial_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_serial_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_serial_k))
        return nk_pack_mismatch_k;

    nk_assert_(nk_packed_shape_matches_(nk_maxsim_packed_shape_, query_packed, query_count, depth) &&
               nk_packed_shape_matches_(nk_maxsim_packed_shape_, document_packed, document_count, depth));
    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_serial_, nk_maxsim_refine_f16_serial_);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_SERIAL

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_MAXSIM_SERIAL_H
