/**
 *  @file include/numkong/spatials/metal.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Batched spatial distances on Metal, launched from C.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_METAL_H
#define NUMKONG_SPATIALS_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/dots/metal.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define nk_define_spatials_metal_(metric, dtype, isa, raw_type, depth_width, per_value, language, alignment)          \
    NUMKONG_API nk_status_t nk_##metric##s_packed_##dtype##_##isa(                                                    \
        nk_cross_##dtype##_operand_t const *a, void const *packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns,    \
        nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {                                \
        nk_size_t bytes;                                                                                              \
        nk_status_t const status = nk_dots_pack_size_##dtype##_##isa(columns, depth, &bytes);                         \
        if (status != nk_success_k) return status;                                                                    \
        if ((nk_size_t)packed & 15) return nk_misaligned_k;                                                           \
        if (bytes > NUMKONG_SIZE_MAX - (nk_size_t)packed) return nk_unexpected_dimensions_k;                          \
        nk_size_t const row_bytes = nk_cross_padded_values_serial_(depth, depth_width, per_value,                     \
                                                                   sizeof(nk_##raw_type##_t)) *                       \
                                    sizeof(nk_##raw_type##_t);                                                        \
        if ((((nk_size_t)packed + sizeof(nk_cross_packed_buffer_header_t)) | row_bytes) & (alignment - 1))            \
            return nk_misaligned_k;                                                                                   \
        nk_size_t const scales_stride = nk_cross_scales_stride_serial_(nk_##dtype##_k, depth);                        \
        nk_cross_operand_t b = {packed, NULL, scales_stride, NULL};                                                   \
        if (nk_block_scaled_format_of_dtype(nk_##dtype##_k).block_size) {                                             \
            b.scales = (nk_u8_t const *)packed + sizeof(nk_cross_packed_buffer_header_t) + columns * row_bytes;       \
            b.tensor_scale = (nk_f32_t const *)((nk_u8_t const *)packed +                                             \
                                                offsetof(nk_cross_packed_buffer_header_t, tensor_scale));             \
        }                                                                                                             \
        return nk_cross_encode_metal_(nk_dots_source_##isa##_, language, nk_cross_threads_##isa##_k,                  \
                                      nk_cross_tile_side_metal_(nk_##dtype##_k, nk_cap_##isa##_k, rows, columns),     \
                                      nk_cross_small_int4_metal_(nk_##dtype##_k, nk_cap_##isa##_k, rows, columns)     \
                                          ? "nk_" #metric "s_" #dtype "_" #isa "_small_kernel_"                       \
                                          : "nk_" #metric "s_" #dtype "_" #isa "_kernel_",                            \
                                      nk_cross_operand_serial_(nk_##dtype##_k, a, a_stride), b,                       \
                                      sizeof(nk_cross_packed_buffer_header_t), c, sizeof(nk_f32_t), 0, rows, columns, \
                                      depth, depth / per_value * sizeof(nk_##raw_type##_t),                           \
                                      columns * (scales_stride + sizeof(nk_f32_t)), a_stride, row_bytes, c_stride, 0, \
                                      nk_cross_scale_blocks_serial_(nk_##dtype##_k, depth), stream);                  \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_##metric##s_symmetric_##dtype##_##isa(                                                 \
        nk_cross_##dtype##_operand_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,       \
        nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {    \
        if (!nk_cross_whole_blocks_serial_(nk_##dtype##_k, depth)) return nk_unexpected_dimensions_k;                 \
        rows_end = nk_min_of_two(rows_end, vector_count);                                                             \
        nk_size_t const window_rows = rows_end > rows_begin ? rows_end - rows_begin : 0;                              \
        nk_cross_operand_t const a = nk_cross_operand_serial_(nk_##dtype##_k, vectors, stride);                       \
        if (depth && (((nk_size_t)a.elements | stride) & (alignment - 1))) return nk_misaligned_k;                    \
        return nk_cross_encode_metal_(                                                                                \
            nk_dots_source_##isa##_, language, nk_cross_threads_##isa##_k,                                            \
            nk_cross_tile_side_metal_(nk_##dtype##_k, nk_cap_##isa##_k, window_rows, vector_count),                   \
            nk_cross_small_int4_metal_(nk_##dtype##_k, nk_cap_##isa##_k, window_rows, vector_count)                   \
                ? "nk_" #metric "s_" #dtype "_" #isa "_small_kernel_"                                                 \
                : "nk_" #metric "s_" #dtype "_" #isa "_kernel_",                                                      \
            a, a, 0, result, sizeof(nk_f32_t), rows_begin, rows_end, vector_count, depth,                             \
            depth / per_value * sizeof(nk_##raw_type##_t), 0, stride, stride, result_stride, 1,                       \
            nk_cross_scale_blocks_serial_(nk_##dtype##_k, depth), stream);                                            \
    }

#if NUMKONG_TARGET_METAL
nk_define_spatials_metal_(angular, i8, metal, i8, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, u8, metal, u8, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, i4, metal, i4x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, u4, metal, u4x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, f16, metal, f16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, bf16, metal, bf16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, e4m3, metal, e4m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, e5m2, metal, e5m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, e3m2, metal, e3m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, e2m3, metal, e2m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, e2m1, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, i8, metal, i8, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, u8, metal, u8, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, i4, metal, i4x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, u4, metal, u4x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, f16, metal, f16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, bf16, metal, bf16, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, e4m3, metal, e4m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, e5m2, metal, e5m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, e3m2, metal, e3m2, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, e2m3, metal, e2m3, 16, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, e2m1, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp8e4m3, metal, e4m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e4m3, metal, e4m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp8e5m2, metal, e5m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp8e5m2, metal, e5m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp6e2m3, metal, e2m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e2m3, metal, e2m3, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp6e3m2, metal, e3m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp6e3m2, metal, e3m2, 32, 1, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, mxfp4, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, mxfp4, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(angular, nvfp4, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
nk_define_spatials_metal_(euclidean, nvfp4, metal, e2m1x2, 32, 2, NUMKONG_METAL_LANGUAGE_3_1_, 1)
#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_SPATIALS_METAL_H
