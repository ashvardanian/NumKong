/**
 *  @file include/numkong/dots/cuda.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief CUDA host side of batched dot products: the pack and tile launches, the generators every
 *      CUDA capability instantiates its exports with, and the baseline @c cuda exports, over the
 *      kernels of `dots/simt.cuh`.
 *
 *  @sa include/numkong/dots/simt.cuh
 *  @sa include/numkong/dots/rocm.cuh
 */
#ifndef NUMKONG_DOTS_CUDA_CUH
#define NUMKONG_DOTS_CUDA_CUH

#include "numkong/cuda.cuh"
#include "numkong/dots/simt.cuh"

#if NUMKONG_ARCH_CUDA_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Launchers

/** Validates the contract and launches as many blocks of @p kernel as stay resident, each walking
 *  @p tile × @p tile output tiles with a stride of the grid. @p b_norms holds the packed column
 *  norms a @c packed metric reads, or is null. @p block_size is the block of a block-scaled dtype,
 *  whose @p depth it must divide and whose operands must carry scales, or zero for plain dtypes.
 *  Codes need 16-byte rows, while scales may sit at any byte, as dense rows of them do. */
NUMKONG_INLINE nk_status_t nk_cross_launch_cuda_(void const *kernel, unsigned tile, unsigned threads,
                                                 nk_cross_operand_t const *a, nk_cross_operand_t const *b,
                                                 void const *b_norms, void *c, nk_size_t result_bytes,
                                                 nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                 nk_size_t depth, nk_size_t block_size, nk_size_t depth_bytes,
                                                 nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                 void *stream) {
    if (block_size && (depth % block_size || !a->scales || !b->scales)) return nk_unexpected_dimensions_k;
    if ((((nk_size_t)a->elements) | a_stride | ((nk_size_t)b->elements) | b_stride) & 15 ||
        (((nk_size_t)c) | c_stride) & (result_bytes - 1))
        return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_size_t const column_tiles = nk_size_divide_round_up_(column_count, tile);
    nk_size_t const tiles = nk_size_divide_round_up_(row_end - row_start, tile) * column_tiles;
    nk_cross_tile_arguments_t arguments;
    arguments.a = (unsigned char const *)a->elements, arguments.b = (unsigned char const *)b->elements;
    arguments.c = c;
    arguments.row_start = row_start, arguments.row_end = row_end, arguments.column_count = column_count;
    arguments.depth = depth, arguments.depth_bytes = depth_bytes, arguments.a_stride = a_stride;
    arguments.b_stride = b_stride;
    arguments.c_stride = c_stride, arguments.column_tiles = column_tiles, arguments.tiles = tiles;
    arguments.depth_slabs = nk_size_divide_round_up_(depth_bytes, 64);
    arguments.b_norms = b_norms;
    arguments.a_scales = a->scales, arguments.b_scales = b->scales;
    arguments.a_scales_stride = a->scales_stride, arguments.b_scales_stride = b->scales_stride;
    arguments.a_tensor_scale = a->tensor_scale, arguments.b_tensor_scale = b->tensor_scale;
    return nk_launch_resident_cuda_(kernel, threads, 0, 0, tiles, &arguments, stream);
}

/** Launches @p kernel with one 32-lane group per packed column, walked with a grid stride, which
 *  records the packing @p capability and the tensor scale of @p b. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_cuda_(void const *kernel, nk_cross_operand_t const *b,
                                                      nk_size_t column_count, nk_size_t depth, nk_size_t depth_bytes,
                                                      nk_size_t b_stride, void *b_packed, nk_size_t columns_begin,
                                                      nk_size_t columns_end, nk_size_t depth_values_padded,
                                                      nk_capability_t capability, nk_size_t scales_stride,
                                                      void *stream) {
    nk_size_t const columns = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const needed = nk_size_divide_round_up_(columns, nk_cross_pack_groups_k);
    nk_size_t const blocks = needed == 0 ? 1 : needed < 65535 ? needed : 65535;
    void const *elements = b->elements;
    nk_u8_t const *scale_values = b->scales;
    nk_size_t scale_values_stride = b->scales_stride;
    nk_f32_t const *tensor_scale = b->tensor_scale;
    void *arguments[14];
    arguments[0] = &elements, arguments[1] = &column_count, arguments[2] = &depth, arguments[3] = &depth_bytes;
    arguments[4] = &b_stride, arguments[5] = &b_packed, arguments[6] = &columns_begin, arguments[7] = &columns_end;
    arguments[8] = &depth_values_padded, arguments[9] = &capability, arguments[10] = &scale_values;
    arguments[11] = &scale_values_stride, arguments[12] = &tensor_scale, arguments[13] = &scales_stride;
    return nk_launch_cuda_(kernel, blocks, nk_cross_pack_groups_k * 32, arguments, 0, stream);
}

#pragma endregion Launchers

#pragma region Cross Macros

/**
 *  @brief Generates a packed-shape accessor copying a device-resident packed buffer's header back.
 *  @sa nk_define_cross_packed_shape_ for the host-resident original.
 */
#define nk_define_cross_packed_shape_cuda_(input_type_name, isa_suffix)                      \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##input_type_name##_##isa_suffix(           \
        void const *b_packed, nk_size_t *columns, nk_size_t *depth, void *stream) {          \
        if ((nk_size_t)b_packed & 15) return nk_misaligned_k;                                \
        nk_cross_packed_buffer_header_t header;                                              \
        nk_status_t const status = nk_read_cuda_(&header, b_packed, sizeof(header), stream); \
        if (status != nk_success_k) return status;                                           \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;         \
        *columns = header.column_count, *depth = header.depth_dimensions;                    \
        return nk_success_k;                                                                 \
    }

/** Generates a pack into the serial layout on the device and its kernel, from
 *  @c nk_define_cross_pack_kernel_simt_. */
#define nk_define_cross_pack_rows_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,     \
                                        norm_value_type, compute_norm_fn, depth_simd_dimensions, dimensions_per_value) \
    nk_define_cross_pack_kernel_simt_(input_type_name, isa_suffix, packed_value_type, load_fn, norm_value_type,        \
                                      compute_norm_fn) NUMKONG_API nk_status_t                                         \
    nk_dots_pack_##input_type_name##_##isa_suffix(                                                                     \
        nk_cross_##input_type_name##_operand_t const *b_operand, nk_size_t column_count, nk_size_t depth,              \
        nk_size_t b_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, void *stream) {            \
        nk_cross_operand_t const b = nk_cross_operand_(nk_##input_type_name##_k, b_operand, b_stride);                 \
        nk_size_t const depth_values_padded = nk_cross_padded_values_simt_(                                            \
            depth, depth_simd_dimensions, dimensions_per_value, sizeof(nk_##packed_value_type##_t));                   \
        nk_size_t const depth_bytes = depth / dimensions_per_value * sizeof(nk_##input_value_type##_t);                \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth);                      \
        if (!nk_cross_whole_blocks_(nk_##input_type_name##_k, depth) || (scales_stride && !b.scales))                  \
            return nk_unexpected_dimensions_k;                                                                         \
        return nk_cross_pack_launch_cuda_((void const *)nk_dots_pack_##input_type_name##_##isa_suffix##_kernel_, &b,   \
                                          column_count, depth, depth_bytes, b_stride, b_packed, columns_begin,         \
                                          columns_end, depth_values_padded, nk_cap_##isa_suffix##_k, scales_stride,    \
                                          stream);                                                                     \
    }

/** The pack's size, shape reader and kernel, which always ship together, with the norm share of
 *  the input type, @c nk_<input_type_name>_lane_sumsq_. */
#define nk_define_cross_pack_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,   \
                                   norm_value_type, depth_simd_dimensions, dimensions_per_value)                \
    nk_define_cross_pack_size_simt_(input_type_name, isa_suffix, packed_value_type, norm_value_type,            \
                                    depth_simd_dimensions, dimensions_per_value)                                \
    nk_define_cross_packed_shape_cuda_(input_type_name, isa_suffix)                                             \
    nk_define_cross_pack_rows_cuda_(input_type_name, isa_suffix, input_value_type, packed_value_type, load_fn,  \
                                    norm_value_type, nk_##input_type_name##_lane_sumsq_, depth_simd_dimensions, \
                                    dimensions_per_value)

/**
 *  @brief Generates C = A × Bᵀ, or its angular or euclidean distances, on @p tile over a B packed
 *      by @c nk_define_cross_pack_cuda_, with its kernel from @c nk_define_cross_packed_kernel_simt_.
 *  @sa nk_define_cross_packed_ for the host original.
 */
#define nk_define_cross_packed_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,  \
                                     result_value_type, depth_simd_dimensions, dimensions_per_value, ...)             \
    nk_define_cross_packed_kernel_simt_(metric, input_type_name, isa_suffix, tile, __VA_ARGS__)                       \
        NUMKONG_API nk_status_t                                                                                       \
        nk_##metric##s_packed_##input_type_name##_##isa_suffix(                                                       \
            nk_cross_##input_type_name##_operand_t const *a_operand, void const *b_packed_buffer,                     \
            nk_##result_value_type##_t *c_matrix, nk_size_t row_count, nk_size_t column_count, nk_size_t depth,       \
            nk_size_t a_stride, nk_size_t c_stride, void *stream) {                                                   \
        nk_size_t const row_bytes = nk_cross_padded_values_simt_(depth, depth_simd_dimensions, dimensions_per_value,  \
                                                                 sizeof(nk_##packed_value_type##_t)) *                \
                                    sizeof(nk_##packed_value_type##_t);                                               \
        nk_size_t const scales_stride = nk_cross_scales_stride_(nk_##input_type_name##_k, depth);                     \
        nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed_buffer;     \
        nk_u8_t const *b_rows = (nk_u8_t const *)(header + 1);                                                        \
        nk_cross_operand_t const a = nk_cross_operand_(nk_##input_type_name##_k, a_operand, a_stride);                \
        nk_cross_operand_t const b = {b_rows, scales_stride ? b_rows + column_count * row_bytes : NUMKONG_NULL,       \
                                      scales_stride, &header->tensor_scale};                                          \
        return nk_cross_launch_cuda_(                                                                                 \
            (void const *)nk_##metric##s_packed_##input_type_name##_##isa_suffix##_kernel_, nk_cross_tile_##tile##_k, \
            nk_cross_threads_##tile##_k, &a, &b, b_rows + column_count * (row_bytes + scales_stride), c_matrix,       \
            sizeof(nk_##result_value_type##_t), 0, row_count, column_count, depth,                                    \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), a_stride, row_bytes, c_stride, stream); \
    }

/**
 *  @brief Generates the Gram matrix C = A × Aᵀ, or its angular or euclidean distances, on @p tile
 *      over rows [row_start, row_start + row_count), with its kernel from
 *      @c nk_define_cross_symmetric_kernel_simt_.
 *
 *  Takes the parameters of @c nk_define_cross_packed_cuda_, so one bundle feeds both.
 *
 *  @sa nk_define_cross_symmetric_ for the host original.
 */
#define nk_define_cross_symmetric_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type,                  \
                                        packed_value_type, result_value_type, depth_simd_dimensions,                  \
                                        dimensions_per_value, ...)                                                    \
    nk_define_cross_symmetric_kernel_simt_(metric, input_type_name, isa_suffix, tile, __VA_ARGS__)                    \
        NUMKONG_API nk_status_t                                                                                       \
        nk_##metric##s_symmetric_##input_type_name##_##isa_suffix(                                                    \
            nk_cross_##input_type_name##_operand_t const *vectors_operand, nk_size_t vectors_count, nk_size_t depth,  \
            nk_size_t stride, nk_##result_value_type##_t *result, nk_size_t result_stride, nk_size_t row_start,       \
            nk_size_t row_count, void *stream) {                                                                      \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;      \
        nk_cross_operand_t const vectors = nk_cross_operand_(nk_##input_type_name##_k, vectors_operand, stride);      \
        return nk_cross_launch_cuda_(                                                                                 \
            (void const *)nk_##metric##s_symmetric_##input_type_name##_##isa_suffix##_kernel_,                        \
            nk_cross_tile_##tile##_k, nk_cross_threads_##tile##_k, &vectors, &vectors, 0, result,                     \
            sizeof(nk_##result_value_type##_t), row_start, row_end, vectors_count, depth,                             \
            nk_block_scaled_format_of_dtype(nk_##input_type_name##_k).block_size,                                     \
            depth / dimensions_per_value * sizeof(nk_##input_value_type##_t), stride, stride, result_stride, stream); \
    }

/** Both shapes of one metric on @p tile, packed and symmetric. */
#define nk_define_cross_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,       \
                              result_value_type, depth_simd_dimensions, dimensions_per_value, ...)                  \
    nk_define_cross_packed_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type,    \
                                 result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)       \
    nk_define_cross_symmetric_cuda_(metric, input_type_name, isa_suffix, tile, input_value_type, packed_value_type, \
                                    result_value_type, depth_simd_dimensions, dimensions_per_value, __VA_ARGS__)

#pragma endregion Cross Macros

#pragma region Baseline Kernels

/*  NVIDIA stages depth in F32 words for its FMA, and folds bytes and nibbles through @c dp4a. */
#if NUMKONG_TARGET_CUDA
nk_define_cross_pack_cuda_(f64, cuda, f64, f64, nk_load_b8_, f64, 2, 1)
nk_define_cross_cuda_(dot, f64, cuda, f64_simt, f64, f64, f64, 2, 1, nk_f64_k, nk_cross_accumulation_dot2_k)
nk_define_cross_pack_cuda_(f32, cuda, f32, f32, nk_load_b8_, f64, 4, 1)
nk_define_cross_cuda_(dot, f32, cuda, f64_simt, f32, f32, f64, 4, 1, nk_f32_k, nk_cross_accumulation_f64_k)
nk_define_cross_pack_cuda_(bf16, cuda, bf16, bf16, nk_load_b8_, f32, 8, 1)
nk_define_cross_cuda_(dot, bf16, cuda, b32_simt, bf16, bf16, f32, 8, 1, nk_bf16_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(f16, cuda, f16, f16, nk_load_b8_, f32, 8, 1)
nk_define_cross_cuda_(dot, f16, cuda, b32_simt, f16, f16, f32, 8, 1, nk_f16_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e5m2, cuda, e5m2, e5m2, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e5m2, cuda, b32_simt, e5m2, e5m2, f32, 16, 1, nk_e5m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e4m3, cuda, e4m3, e4m3, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e4m3, cuda, b32_simt, e4m3, e4m3, f32, 16, 1, nk_e4m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e3m2, cuda, e3m2, e3m2, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e3m2, cuda, b32_simt, e3m2, e3m2, f32, 16, 1, nk_e3m2_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e2m3, cuda, e2m3, e2m3, nk_load_b8_, f32, 16, 1)
nk_define_cross_cuda_(dot, e2m3, cuda, b32_simt, e2m3, e2m3, f32, 16, 1, nk_e2m3_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(e2m1, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, 32, 2)
nk_define_cross_cuda_(dot, e2m1, cuda, b32_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_e2m1_k, nk_cross_accumulation_f32_k)
nk_define_cross_pack_cuda_(i8, cuda, i8, i8, nk_load_b8_, u32, 16, 1)
nk_define_cross_cuda_(dot, i8, cuda, b32_simt, i8, i8, i32, 16, 1, nk_i8_k, nk_cross_accumulation_i8x4_k)
nk_define_cross_pack_cuda_(u8, cuda, u8, u8, nk_load_b8_, u32, 16, 1)
nk_define_cross_cuda_(dot, u8, cuda, b32_simt, u8, u8, u32, 16, 1, nk_u8_k, nk_cross_accumulation_u8x4_k)
nk_define_cross_pack_cuda_(i4, cuda, i4x2, i4x2, nk_load_b8_, u32, 32, 2)
nk_define_cross_cuda_(dot, i4, cuda, b32_simt, i4x2, i4x2, i32, 32, 2, nk_i4_k, nk_cross_accumulation_i4x4_k)
nk_define_cross_pack_cuda_(u4, cuda, u4x2, u4x2, nk_load_b8_, u32, 32, 2)
nk_define_cross_cuda_(dot, u4, cuda, b32_simt, u4x2, u4x2, u32, 32, 2, nk_u4_k, nk_cross_accumulation_u4x4_k)
nk_define_cross_pack_size_simt_(nvfp4, cuda, e2m1x2, f32, 32, 2)
nk_define_cross_packed_shape_cuda_(nvfp4, cuda)
nk_define_cross_pack_rows_cuda_(nvfp4, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_cuda_(dot, nvfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_nvfp4_k)
nk_define_cross_pack_size_simt_(mxfp4, cuda, e2m1x2, f32, 32, 2)
nk_define_cross_packed_shape_cuda_(mxfp4, cuda)
nk_define_cross_pack_rows_cuda_(mxfp4, cuda, e2m1x2, e2m1x2, nk_load_b8_, f32, nk_e2m1_lane_sumsq_, 32, 2)
nk_define_cross_cuda_(dot, mxfp4, cuda, scaled_simt, e2m1x2, e2m1x2, f32, 32, 2, nk_mxfp4_k)
nk_define_cross_pack_size_simt_(mxfp8e4m3, cuda, e4m3, f32, 16, 1)
nk_define_cross_packed_shape_cuda_(mxfp8e4m3, cuda)
nk_define_cross_pack_rows_cuda_(mxfp8e4m3, cuda, e4m3, e4m3, nk_load_b8_, f32, nk_e4m3_lane_sumsq_, 16, 1)
nk_define_cross_cuda_(dot, mxfp8e4m3, cuda, scaled_simt, e4m3, e4m3, f32, 16, 1, nk_mxfp8e4m3_k)
nk_define_cross_pack_size_simt_(mxfp8e5m2, cuda, e5m2, f32, 16, 1)
nk_define_cross_packed_shape_cuda_(mxfp8e5m2, cuda)
nk_define_cross_pack_rows_cuda_(mxfp8e5m2, cuda, e5m2, e5m2, nk_load_b8_, f32, nk_e5m2_lane_sumsq_, 16, 1)
nk_define_cross_cuda_(dot, mxfp8e5m2, cuda, scaled_simt, e5m2, e5m2, f32, 16, 1, nk_mxfp8e5m2_k)
#endif // NUMKONG_TARGET_CUDA

#pragma endregion Baseline Kernels

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_CUDA_
#endif // NUMKONG_DOTS_CUDA_CUH
