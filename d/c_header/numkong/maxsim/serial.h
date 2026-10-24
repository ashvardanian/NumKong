/**
 *  @brief SWAR-accelerated MaxSim (ColBERT late-interaction) for SIMD-free CPUs.
 *  @file include/numkong/maxsim/serial.h
 *  @author Ash Vardanian
 *  @date February 17, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Defines the packed buffer header and per-vector metadata structures used by all MaxSim ISA backends,
 *  plus scalar reference implementations for correctness validation.
 *
 *  MaxSim computes: result = Σᵢ minⱼ angular(qᵢ, dⱼ) — angular distance late-interaction scoring.
 *
 *  Strategy: coarse i8-quantized screening with running argmax (dot as proxy for argmin angular),
 *  then full-precision refinement of the winning (query, document) pairs via existing nk_dot_* primitives,
 *  finalized with angular distance: 1 - dot / sqrt(||q||² × ||d||²).
 *
 *  @section packed_layout Packed Buffer Layout
 *
 *  [Header 64B] [i8 vectors, 64B-aligned] [metadata, 64B-aligned] [originals row-major, 64B-aligned]
 *
 *  - i8 region: row-major with padded depth for SIMD alignment
 *  - Metadata region: vector_count x 12 bytes (scale + sum + norm_squared per vector)
 *  - Originals region: row-major bf16 or f32, stride padded to 64B for nk_dot_* calls
 */
#ifndef NK_MAXSIM_SERIAL_H
#define NK_MAXSIM_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_bf16_to_f32_serial`
#include "numkong/dot.h"            // `nk_dot_bf16`, `nk_dot_f32`, `nk_dot_f16`
#include "numkong/spatial/serial.h" // `nk_f32_rsqrt_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Packed buffer header (64 bytes, cache-line aligned).
 *  Stored at the beginning of every maxsim packed buffer.
 */
typedef struct {
    nk_u32_t vector_count;           ///< Number of vectors packed
    nk_u32_t depth_dimensions;       ///< Logical depth (number of elements per vector)
    nk_u32_t depth_i8_padded;        ///< Padded i8 depth in bytes (SIMD-aligned)
    nk_u32_t original_element_bytes; ///< 2 for bf16, 4 for f32
    nk_u32_t offset_i8_data;         ///< Byte offset from buffer start to i8 region
    nk_u32_t offset_metadata;        ///< Byte offset from buffer start to metadata region
    nk_u32_t offset_original_data;   ///< Byte offset from buffer start to originals region
    nk_u32_t original_stride_bytes;  ///< Row stride in bytes for originals region
    nk_u32_t reserved[8];            ///< Padding to 64 bytes
} nk_maxsim_packed_header_t;

NK_STATIC_ASSERT(sizeof(nk_maxsim_packed_header_t) == 64, nk_maxsim_packed_header_must_be_64_bytes);

/**
 *  @brief Per-vector quantization metadata (12 bytes).
 *  Stored in the metadata region of the packed buffer, one per vector.
 */
typedef struct {
    nk_f32_t scale_f32;        ///< Quantization scale: absmax / range_limit
    nk_i32_t sum_i8_i32;       ///< Sum of all i8 quantized elements (for VPDPBUSD/VPMADDUBSW bias correction)
    nk_f32_t inverse_norm_f32; ///< 1/sqrt(||v||^2), 0 if zero-vector — precomputed for angular finalization
} nk_maxsim_vector_metadata_t;

NK_STATIC_ASSERT(sizeof(nk_maxsim_vector_metadata_t) == 12, nk_maxsim_vector_metadata_must_be_12_bytes);

/**
 *  @brief Conversion function pointer type for element-to-f32 conversion.
 *  Each conversion reads one element from `source` and writes one f32 to `destination`.
 */
typedef void (*nk_maxsim_to_f32_t)(void const *source, nk_f32_t *destination);

/** @brief Identity conversion for f32 sources — just a typed memcpy. */
NK_INTERNAL void nk_f32_to_f32_(void const *source, nk_f32_t *destination) { *destination = *(nk_f32_t const *)source; }

/**
 *  @brief Fills the packed buffer header and returns the padded i8 depth.
 *  Consolidates header/offset computation duplicated in every pack function.
 */
NK_INTERNAL nk_size_t nk_maxsim_packed_header_setup_(      //
    void *packed, nk_size_t vector_count, nk_size_t depth, //
    nk_size_t depth_simd_dimensions, nk_size_t original_element_bytes) {

    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += depth_simd_dimensions;

    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(vector_count * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        vector_count * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * original_element_bytes, 64);

    nk_maxsim_packed_header_t *header = (nk_maxsim_packed_header_t *)packed;
    header->vector_count = (nk_u32_t)vector_count;
    header->depth_dimensions = (nk_u32_t)depth;
    header->depth_i8_padded = (nk_u32_t)depth_i8_padded;
    header->original_element_bytes = (nk_u32_t)original_element_bytes;
    header->offset_i8_data = (nk_u32_t)header_size;
    header->offset_metadata = (nk_u32_t)(header_size + i8_region_size);
    header->offset_original_data = (nk_u32_t)(header_size + i8_region_size + metadata_region_size);
    header->original_stride_bytes = (nk_u32_t)original_stride;
    for (nk_size_t reserved_index = 0; reserved_index < 8; reserved_index++) header->reserved[reserved_index] = 0;

    return depth_i8_padded;
}

/**
 *  @brief Quantizes a single source vector to i8, computing metadata.
 *  Iterates element-by-element, calling the conversion callback for each f32 value.
 *  No temp buffer needed — works for arbitrary depth.
 */
NK_INTERNAL void nk_maxsim_quantize_vector_(                             //
    void const *source_vector, nk_size_t element_bytes, nk_size_t depth, //
    nk_size_t depth_i8_padded, nk_f32_t scale_limit,                     //
    nk_maxsim_to_f32_t convert_to_f32,                                   //
    nk_i8_t *destination_i8, nk_maxsim_vector_metadata_t *metadata,      //
    nk_f32_t *norm_squared_ptr) {

    char const *source_bytes = (char const *)source_vector;

    // Pass 1: Find absmax, compute norm_squared
    nk_f32_t absmax_f32 = 0.0f;
    nk_f32_t norm_squared_f32 = 0.0f;
    for (nk_size_t dim_index = 0; dim_index < depth; dim_index++) {
        nk_f32_t value_f32;
        convert_to_f32(source_bytes + dim_index * element_bytes, &value_f32);
        nk_f32_t abs_value = nk_f32_abs_(value_f32);
        if (abs_value > absmax_f32) absmax_f32 = abs_value;
        norm_squared_f32 += value_f32 * value_f32;
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

    metadata->scale_f32 = scale_f32;
    metadata->sum_i8_i32 = sum_quantized_i32;
    *norm_squared_ptr = norm_squared_f32;
}

/**
 *  @brief Region pointers extracted from two packed buffers.
 *  Eliminates ~15 lines of boilerplate per compute function.
 */
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

NK_INTERNAL nk_maxsim_packed_regions_t nk_maxsim_extract_packed_regions_( //
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
 *  Layout: header + i8 data (64B-aligned) + metadata (64B-aligned) + originals (64B-aligned)
 *
 *  @param vector_count Number of vectors to pack.
 *  @param depth Number of elements per vector.
 *  @param original_element_bytes Size of each original element (2 for bf16, 4 for f32).
 *  @param depth_simd_dimensions SIMD width for i8 depth padding (1 for serial).
 */
NK_INTERNAL nk_size_t nk_maxsim_packed_size_( //
    nk_size_t vector_count, nk_size_t depth,  //
    nk_size_t original_element_bytes, nk_size_t depth_simd_dimensions) {

    // Step 1: Pad i8 depth to SIMD width
    nk_size_t depth_i8_padded = nk_size_round_up_to_multiple_(depth, depth_simd_dimensions);

    // Step 2: Break power-of-2 strides for cache associativity
    if ((depth_i8_padded & (depth_i8_padded - 1)) == 0 && depth_i8_padded > 0) depth_i8_padded += depth_simd_dimensions;

    // Step 3: Calculate region sizes
    nk_size_t const header_size = sizeof(nk_maxsim_packed_header_t);
    nk_size_t const i8_region_size = nk_size_round_up_to_multiple_(vector_count * depth_i8_padded, 64);
    nk_size_t const metadata_region_size = nk_size_round_up_to_multiple_(
        vector_count * sizeof(nk_maxsim_vector_metadata_t), 64);
    nk_size_t const original_stride = nk_size_round_up_to_multiple_(depth * original_element_bytes, 64);
    nk_size_t const originals_region_size = vector_count * original_stride;

    return header_size + i8_region_size + metadata_region_size + originals_region_size;
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_serial(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_bf16_t), 1);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_serial(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f32_t), 1);
}

NK_PUBLIC void nk_maxsim_pack_bf16_serial( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_serial,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? nk_f32_rsqrt_serial(norm_sq) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f32_serial( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? nk_f32_rsqrt_serial(norm_sq) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_serial(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f16_t), 1);
}

NK_PUBLIC void nk_maxsim_pack_f16_serial( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 1, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_serial,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? nk_f32_rsqrt_serial(norm_sq) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/**
 *  @brief Dtype-agnostic coarse i8 argmax kernel for the serial backend.
 *  Produces per-query best document indices using signed i8×i8 dot products.
 *  No bias correction needed — serial uses native signed×signed multiplication.
 */
NK_INTERNAL void nk_maxsim_coarse_argmax_serial_( //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth_i8_padded, nk_u32_t *best_document_indices) {

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        nk_i32_t running_max_i32[4] = {NK_I32_MIN, NK_I32_MIN, NK_I32_MIN, NK_I32_MIN};
        nk_u32_t running_argmax_u32[4] = {0, 0, 0, 0};

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

            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++) {
                if (accumulator_i32[query_tile_index] > running_max_i32[query_tile_index]) {
                    running_max_i32[query_tile_index] = accumulator_i32[query_tile_index];
                    running_argmax_u32[query_tile_index] = (nk_u32_t)document_index;
                }
            }
        }

        for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
            best_document_indices[query_block_start_index + query_tile_index] = running_argmax_u32[query_tile_index];
    }

    // Edge path: remaining 1-3 queries
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            nk_i32_t accumulator_i32 = 0;

            for (nk_size_t dim_index = 0; dim_index < depth_i8_padded; dim_index++)
                accumulator_i32 += (nk_i32_t)query_i8_row[dim_index] * (nk_i32_t)document_i8_row[dim_index];

            if (accumulator_i32 > running_max_i32) {
                running_max_i32 = accumulator_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        best_document_indices[query_index] = running_argmax_u32;
    }
}

NK_PUBLIC void nk_maxsim_packed_bf16_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_serial_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                        regions.document_quantized, chunk_size, document_count, regions.depth_i8_padded,
                                        best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_bf16((nk_bf16_t const *)(regions.query_originals +
                                            (chunk_start + query_index) * regions.query_original_stride),
                        (nk_bf16_t const *)(regions.document_originals +
                                            best_document_index * regions.document_original_stride),
                        depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

NK_PUBLIC void nk_maxsim_packed_f32_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_serial_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                        regions.document_quantized, chunk_size, document_count, regions.depth_i8_padded,
                                        best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f64_t dot_result;
            nk_dot_f32(
                (nk_f32_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f32_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f64_t cosine = dot_result *
                              (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f64_t angular = 1.0 - cosine;
            if (angular < 0.0) angular = 0.0;
            total_angular_distance += angular;
        }
    }

    *result = total_angular_distance;
}

NK_PUBLIC void nk_maxsim_packed_f16_serial( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_serial_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                        regions.document_quantized, chunk_size, document_count, regions.depth_i8_padded,
                                        best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_f16(
                (nk_f16_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f16_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_MAXSIM_SERIAL_H
