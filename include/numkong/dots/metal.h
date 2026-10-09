/**
 *  @file include/numkong/dots/metal.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every Apple GPU, launched from C.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/metal.metal, the kernels this embeds
 *  @sa include/numkong/dots/simt.cuh, the CUDA and ROCm sibling
 *
 *  Every signature matches the @c cuda capability's, with an @c id<MTLCommandQueue> as the stream.
 *  The kernels travel as their own `.metal` source, embedded here and compiled on the device at
 *  first use under Metal 3.1, the first with @c bfloat, so every Apple GPU from family 7 runs them.
 *  The pack, its shape reader and the launch core serve @c apple9 and @c apple10 too.
 */
#ifndef NUMKONG_DOTS_METAL_H
#define NUMKONG_DOTS_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h"       // `nk_metal_call_t`
#include "numkong/dots/serial.h" // `nk_cross_packed_buffer_header_t`, `nk_cross_padded_values_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_dots_source_metal_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/dots/metal.metal" suffix(, )
#embed "numkong/spatials/metal.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Launchers

/** Side of the output tile one threadgroup owns, and the threads it runs, as in the kernels. */
enum { nk_cross_tile_metal_k = 64, nk_cross_threads_metal_k = 256 };

/** The launch record of a tile kernel, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t a_stride, b_stride, c_stride;
    nk_u32_t rows_begin, rows_end, column_count, depth;
    nk_u32_t upper_triangle;
    nk_u64_t a_scales_stride, b_scales_stride;
    nk_u32_t a_tensor_present, b_tensor_present;
} nk_cross_arguments_metal_t;

/** The launch record of a pack kernel, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t b_stride, depth_bytes, row_bytes;
    nk_capability_t capability;
    nk_u32_t column_count, depth, depth_padded_values, columns_begin, columns_end;
    nk_u64_t scales_stride;
    nk_u32_t tensor_present;
} nk_cross_pack_arguments_metal_t;

nk_static_assert_(sizeof(nk_cross_arguments_metal_t) == 72, nk_cross_arguments_metal_must_be_72_bytes);
nk_static_assert_(sizeof(nk_cross_pack_arguments_metal_t) == 72, nk_cross_pack_arguments_metal_must_be_72_bytes);

NUMKONG_INLINE int nk_cross_small_int4_metal_(nk_dtype_t dtype, nk_capability_t capability, nk_size_t rows,
                                              nk_size_t columns) {
    if (capability != nk_cap_apple10_k || (dtype != nk_i4_k && dtype != nk_u4_k)) return 0;
    nk_size_t const row_tiles = rows / 64 + (rows % 64 != 0), column_tiles = columns / 64 + (columns % 64 != 0);
    return row_tiles < 32 && column_tiles < 32 && row_tiles * column_tiles < 32;
}

NUMKONG_INLINE nk_size_t nk_cross_tile_side_metal_(nk_dtype_t dtype, nk_capability_t capability, nk_size_t rows,
                                                   nk_size_t columns) {
    if (nk_cross_small_int4_metal_(dtype, capability, rows, columns)) return 32;
    return capability != nk_cap_metal_k && nk_block_scaled_format_of_dtype(dtype).block_size ? 32 : 64;
}

/** Bytes of a packed B: the header, padded code and scale rows, and one norm per column. */
NUMKONG_INLINE nk_size_t nk_cross_pack_size_metal_(nk_size_t columns, nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                   nk_size_t value_bytes, nk_size_t norm_bytes,
                                                   nk_size_t dimensions_per_value, nk_size_t scales_stride) {
    if (columns > 0xFFFFFFFFu || depth > 0xFFFFFFFFu || !depth_simd_dimensions || !dimensions_per_value) return 0;
    nk_size_t const values = nk_cross_padded_values_serial_(depth, depth_simd_dimensions, dimensions_per_value,
                                                            value_bytes);
    if (values > 0xFFFFFFFFu) return 0;
    nk_size_t row_bytes, body_bytes, norms_bytes;
    if (!nk_size_mul_checked_(values, value_bytes, &row_bytes) || scales_stride > NUMKONG_SIZE_MAX - row_bytes ||
        !nk_size_mul_checked_(columns, row_bytes + scales_stride, &body_bytes) ||
        !nk_size_mul_checked_(columns, norm_bytes, &norms_bytes) ||
        body_bytes > NUMKONG_SIZE_MAX - sizeof(nk_cross_packed_buffer_header_t) ||
        norms_bytes > NUMKONG_SIZE_MAX - sizeof(nk_cross_packed_buffer_header_t) - body_bytes)
        return 0;
    return sizeof(nk_cross_packed_buffer_header_t) + body_bytes + norms_bytes;
}

/** Synchronizes, then reads the columns and depth out of a packed B's header, if @p capability
 *  packed it. */
NUMKONG_INLINE nk_status_t nk_cross_packed_shape_metal_(void const *b_packed, nk_size_t *columns, nk_size_t *depth,
                                                        nk_capability_t capability, nk_stream_t stream) {
    if ((nk_size_t)b_packed & 15) return nk_misaligned_k;
    nk_status_t const status = nk_stream_synchronize_metal_(stream);
    if (status != nk_success_k) return status;
    nk_metal_call_t call;
    nk_status_t entered = nk_enter_metal_(stream, &call);
    if (entered != nk_success_k) return entered;
    nk_size_t offset;
    if (!nk_resolve_metal_(&call, b_packed, sizeof(nk_cross_packed_buffer_header_t), &offset))
        return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != capability) return nk_abort_metal_(&call, nk_pack_mismatch_k);
    *columns = header->column_count, *depth = header->depth_dimensions;
    return nk_abort_metal_(&call, nk_success_k);
}

/** Encodes one pack of columns [begin,end), one simdgroup per column, recording the packing
 *  @p capability. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_metal_(char const *kernel, nk_cross_operand_t b, nk_size_t columns,
                                                       nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                       nk_size_t columns_begin, nk_size_t columns_end,
                                                       nk_size_t depth_simd_dimensions, nk_size_t value_bytes,
                                                       nk_size_t norm_bytes, nk_size_t dimensions_per_value,
                                                       nk_size_t scale_blocks, nk_capability_t capability,
                                                       nk_stream_t stream) {
    if ((nk_size_t)b_packed & 15) return nk_misaligned_k;
    if (columns > 0xFFFFFFFFu || depth > 0xFFFFFFFFu || !dimensions_per_value || depth % dimensions_per_value)
        return nk_unexpected_dimensions_k;
    if (columns_end > columns) columns_end = columns;
    if (columns_end <= columns_begin && columns_begin != 0) return nk_success_k;
    if (scale_blocks && columns_end > columns_begin && !b.scales) return nk_device_memory_mismatch_k;
    nk_metal_call_t call;
    nk_status_t status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t b_offset = 0, packed_offset = 0, scales_offset = 0, tensor_offset = 0;
    nk_size_t row_bytes, b_bytes;
    nk_size_t const scales_stride = scale_blocks ? nk_size_round_up_to_multiple_(scale_blocks, 16) : 0;
    if (scales_stride > 0xFFFFFFFFu) return nk_abort_metal_(&call, nk_unexpected_dimensions_k);
    nk_size_t const packed_bytes = nk_cross_pack_size_metal_(columns, depth, depth_simd_dimensions, value_bytes,
                                                             norm_bytes, dimensions_per_value, scales_stride);
    if (!packed_bytes || !nk_size_mul_checked_(depth / dimensions_per_value, value_bytes, &row_bytes) ||
        !nk_size_span_checked_(row_bytes ? columns_end : 0, b_stride, row_bytes, &b_bytes))
        return nk_abort_metal_(&call, nk_unexpected_dimensions_k);
    nk_size_t scales_bytes;
    if (!nk_size_span_checked_(scale_blocks ? columns_end : 0, b.scales_stride, scale_blocks, &scales_bytes))
        return nk_abort_metal_(&call, nk_unexpected_dimensions_k);
    void *const b_buffer = b_bytes && columns_end > columns_begin
                               ? nk_resolve_metal_(&call, b.elements, b_bytes, &b_offset)
                               : NULL;
    void *const scales_buffer = b.scales && scales_bytes
                                    ? nk_resolve_metal_(&call, b.scales, scales_bytes, &scales_offset)
                                    : NULL;
    void *const tensor_buffer = b.tensor_scale
                                    ? nk_resolve_metal_(&call, b.tensor_scale, sizeof(nk_f32_t), &tensor_offset)
                                    : NULL;
    void *const packed_buffer = nk_resolve_metal_(&call, b_packed, packed_bytes, &packed_offset);
    if (!packed_buffer || (!b_buffer && b_bytes && columns_end > columns_begin) ||
        (b.scales && scales_bytes && !scales_buffer) || (b.tensor_scale && !tensor_buffer))
        return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, nk_dots_source_metal_, kernel, NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    nk_cross_pack_arguments_metal_t arguments;
    nk_size_t const values = nk_cross_padded_values_serial_(depth, depth_simd_dimensions, dimensions_per_value,
                                                            value_bytes);
    arguments.b_stride = b_stride, arguments.depth_bytes = depth / dimensions_per_value * value_bytes;
    arguments.row_bytes = values * value_bytes;
    arguments.column_count = (nk_u32_t)columns, arguments.depth = (nk_u32_t)depth;
    arguments.depth_padded_values = (nk_u32_t)values;
    arguments.columns_begin = (nk_u32_t)columns_begin, arguments.columns_end = (nk_u32_t)columns_end;
    arguments.capability = capability;
    arguments.scales_stride = b.scales_stride;
    arguments.tensor_present = b.tensor_scale != NULL;

    // An empty window still writes the header when it starts at column zero, so one group runs.
    nk_size_t const simdgroups_per_group = 8;
    nk_size_t const columns_count = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const groups = columns_count ? nk_size_divide_round_up_(columns_count, simdgroups_per_group) : 1;
    nk_bind_metal_(call.encoder, b_buffer ? b_buffer : packed_buffer, b_buffer ? b_offset : packed_offset, 0);
    nk_bind_metal_(call.encoder, packed_buffer, packed_offset, 1);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 2);
    nk_bind_metal_(call.encoder, scales_buffer ? scales_buffer : packed_buffer,
                   scales_buffer ? scales_offset : packed_offset, 3);
    nk_bind_metal_(call.encoder, tensor_buffer ? tensor_buffer : packed_buffer,
                   tensor_buffer ? tensor_offset : packed_offset, 4);
    nk_metal_size_t const grid = {groups, 1, 1}, threads = {simdgroups_per_group * 32, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/** Validates the contract and encodes C = A × Bᵀ for the rows from @p rows_begin up to @p rows_end,
 *  one threadgroup per square tile of @p tile_side elements, with A and B strides in bytes. */
NUMKONG_INLINE nk_status_t nk_cross_encode_metal_(char const *source, nk_size_t language_version, nk_size_t threads,
                                                  nk_size_t tile_side, char const *kernel, nk_cross_operand_t a,
                                                  nk_cross_operand_t b, nk_size_t b_extra_offset, void *c,
                                                  nk_size_t result_bytes, nk_size_t rows_begin, nk_size_t rows_end,
                                                  nk_size_t column_count, nk_size_t depth, nk_size_t input_row_bytes,
                                                  nk_size_t b_tail_bytes, nk_size_t a_stride, nk_size_t b_stride,
                                                  nk_size_t c_stride, nk_u32_t upper_triangle, nk_size_t scale_blocks,
                                                  nk_stream_t stream) {
    if ((input_row_bytes && !scale_blocks && ((((nk_size_t)a.elements) | a_stride) & 15)) ||
        ((((nk_size_t)c) | c_stride) & (result_bytes - 1)))
        return nk_misaligned_k;
    if (rows_end <= rows_begin || column_count == 0) return nk_success_k;
    if (scale_blocks && (!a.scales || !b.scales)) return nk_device_memory_mismatch_k;
    if (rows_end > 0xFFFFFFFFu || column_count > 0xFFFFFFFFu || depth > 0xFFFFFFFFu) return nk_unexpected_dimensions_k;
    nk_metal_call_t call;
    nk_status_t status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t a_offset = 0, b_offset = 0, c_offset = 0;
    nk_size_t a_bytes, b_bytes, c_bytes, result_row_bytes;
    if (!nk_size_span_checked_(input_row_bytes ? rows_end : 0, a_stride, input_row_bytes, &a_bytes) ||
        !nk_size_span_checked_(input_row_bytes ? column_count : 0, b_stride,
                               b_extra_offset ? b_stride : input_row_bytes, &b_bytes) ||
        !nk_size_mul_checked_(column_count, result_bytes, &result_row_bytes) ||
        !nk_size_span_checked_(rows_end, c_stride, result_row_bytes, &c_bytes) ||
        b_bytes > NUMKONG_SIZE_MAX - b_extra_offset || b_tail_bytes > NUMKONG_SIZE_MAX - b_extra_offset - b_bytes)
        return nk_abort_metal_(&call, nk_unexpected_dimensions_k);
    b_bytes += b_extra_offset + b_tail_bytes;
    void *const c_buffer = nk_resolve_metal_(&call, c, c_bytes, &c_offset);
    void *const a_buffer = a_bytes ? nk_resolve_metal_(&call, a.elements, a_bytes, &a_offset) : c_buffer;
    void *const b_buffer = b_bytes ? nk_resolve_metal_(&call, b.elements, b_bytes, &b_offset) : c_buffer;
    if (!a_bytes) a_offset = c_offset;
    if (!b_bytes) b_offset = c_offset;
    if (!a_buffer || !b_buffer || !c_buffer) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    nk_size_t a_scales_bytes, b_scales_bytes;
    if (!nk_size_span_checked_(scale_blocks ? rows_end : 0, a.scales_stride, scale_blocks, &a_scales_bytes) ||
        !nk_size_span_checked_(scale_blocks ? column_count : 0, b.scales_stride, scale_blocks, &b_scales_bytes))
        return nk_abort_metal_(&call, nk_unexpected_dimensions_k);
    nk_size_t a_scales_offset = 0, b_scales_offset = 0, a_tensor_offset = 0, b_tensor_offset = 0;
    void *const a_scales_buffer = a.scales && a_scales_bytes
                                      ? nk_resolve_metal_(&call, a.scales, a_scales_bytes, &a_scales_offset)
                                      : NULL;
    void *const b_scales_buffer = b.scales && b_scales_bytes
                                      ? nk_resolve_metal_(&call, b.scales, b_scales_bytes, &b_scales_offset)
                                      : NULL;
    void *const a_tensor_buffer = a.tensor_scale
                                      ? nk_resolve_metal_(&call, a.tensor_scale, sizeof(nk_f32_t), &a_tensor_offset)
                                      : NULL;
    void *const b_tensor_buffer = b.tensor_scale
                                      ? nk_resolve_metal_(&call, b.tensor_scale, sizeof(nk_f32_t), &b_tensor_offset)
                                      : NULL;
    if ((a.scales && a_scales_bytes && !a_scales_buffer) || (b.scales && b_scales_bytes && !b_scales_buffer) ||
        (a.tensor_scale && !a_tensor_buffer) || (b.tensor_scale && !b_tensor_buffer))
        return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, source, kernel, language_version);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    nk_cross_arguments_metal_t arguments;
    arguments.a_stride = a_stride, arguments.b_stride = b_stride, arguments.c_stride = c_stride / result_bytes;
    arguments.rows_begin = (nk_u32_t)rows_begin, arguments.rows_end = (nk_u32_t)rows_end;
    arguments.column_count = (nk_u32_t)column_count, arguments.depth = (nk_u32_t)depth;
    arguments.upper_triangle = upper_triangle;
    arguments.a_scales_stride = a.scales_stride, arguments.b_scales_stride = b.scales_stride;
    arguments.a_tensor_present = a.tensor_scale != NULL, arguments.b_tensor_present = b.tensor_scale != NULL;
    nk_bind_metal_(call.encoder, a_buffer, a_offset, 0);
    nk_bind_metal_(call.encoder, b_buffer, b_offset + b_extra_offset, 1);
    nk_bind_metal_(call.encoder, c_buffer, c_offset, 2);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 3);
    nk_bind_metal_(call.encoder, a_scales_buffer ? a_scales_buffer : a_buffer,
                   a_scales_buffer ? a_scales_offset : a_offset, 4);
    nk_bind_metal_(call.encoder, b_scales_buffer ? b_scales_buffer : b_buffer,
                   b_scales_buffer ? b_scales_offset : b_offset, 5);
    nk_bind_metal_(call.encoder, a_tensor_buffer ? a_tensor_buffer : a_buffer,
                   a_tensor_buffer ? a_tensor_offset : a_offset, 6);
    nk_bind_metal_(call.encoder, b_tensor_buffer ? b_tensor_buffer : b_buffer,
                   b_tensor_buffer ? b_tensor_offset : b_offset, 7);
    nk_metal_size_t const grid = {nk_size_divide_round_up_(column_count, tile_side),
                                  nk_size_divide_round_up_(rows_end - rows_begin, tile_side), 1};
    nk_metal_size_t const group = {threads, 1, 1};
    return nk_dispatch_metal_(&call, grid, group);
}

/** @ref nk_cross_encode_metal_ over the @c metal capability's kernels. */
NUMKONG_INLINE nk_status_t nk_cross_launch_metal_(char const *kernel, nk_cross_operand_t a, nk_cross_operand_t b,
                                                  nk_size_t b_extra_offset, void *c, nk_size_t result_bytes,
                                                  nk_size_t rows_begin, nk_size_t rows_end, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t input_row_bytes, nk_size_t b_tail_bytes,
                                                  nk_size_t a_stride, nk_size_t b_stride, nk_size_t c_stride,
                                                  nk_u32_t upper_triangle, nk_size_t tile_side, nk_size_t scale_blocks,
                                                  nk_stream_t stream) {
    return nk_cross_encode_metal_(nk_dots_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cross_threads_metal_k,
                                  tile_side, kernel, a, b, b_extra_offset, c, result_bytes, rows_begin, rows_end,
                                  column_count, depth, input_row_bytes, b_tail_bytes, a_stride, b_stride, c_stride,
                                  upper_triangle, scale_blocks, stream);
}

#pragma endregion Launchers

#pragma region Cross Macros

/** Every entry of one dtype on the Metal capability @p isa_suffix: the @c metal pack kernel with
 *  rows padded to @p depth_simd_dimensions, and one tile kernel serving both the packed and the
 *  symmetric entries, encoded by that capability's launcher. */
#define nk_define_cross_metal_(dtype, isa, raw_type, result_type, norm_type, depth_width, per_value)                  \
    NUMKONG_API nk_status_t nk_dots_pack_size_##dtype##_##isa(nk_size_t columns, nk_size_t depth, nk_size_t *bytes) { \
        if (!nk_cross_whole_blocks_serial_(nk_##dtype##_k, depth)) return nk_unexpected_dimensions_k;                 \
        *bytes = nk_cross_pack_size_metal_(columns, depth, depth_width, sizeof(nk_##raw_type##_t),                    \
                                           sizeof(nk_##norm_type##_t), per_value,                                     \
                                           nk_cross_scales_stride_serial_(nk_##dtype##_k, depth));                    \
        return *bytes ? nk_success_k : nk_unexpected_dimensions_k;                                                    \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##dtype##_##isa(void const *b, nk_size_t *columns, nk_size_t *depth, \
                                                                 nk_stream_t stream) {                                \
        return nk_cross_packed_shape_metal_(b, columns, depth, nk_cap_##isa##_k, stream);                             \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_dots_pack_##dtype##_##isa(nk_cross_##dtype##_operand_t const *b, nk_size_t columns,    \
                                                         nk_size_t depth, nk_size_t stride, void *packed,             \
                                                         nk_size_t begin, nk_size_t end, nk_stream_t stream) {        \
        if (!nk_cross_whole_blocks_serial_(nk_##dtype##_k, depth)) return nk_unexpected_dimensions_k;                 \
        return nk_cross_pack_launch_metal_(                                                                           \
            "nk_dots_pack_" #dtype "_metal_kernel_", nk_cross_operand_serial_(nk_##dtype##_k, b, stride), columns,    \
            depth, stride, packed, begin, end, depth_width, sizeof(nk_##raw_type##_t), sizeof(nk_##norm_type##_t),    \
            per_value, nk_cross_scale_blocks_serial_(nk_##dtype##_k, depth), nk_cap_##isa##_k, stream);               \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_dots_packed_##dtype##_##isa(                                                           \
        nk_cross_##dtype##_operand_t const *a, void const *packed, nk_##result_type##_t *c, nk_size_t rows,           \
        nk_size_t columns, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {             \
        nk_size_t bytes;                                                                                              \
        nk_status_t const status = nk_dots_pack_size_##dtype##_##isa(columns, depth, &bytes);                         \
        if (status != nk_success_k) return status;                                                                    \
        if ((nk_size_t)packed & 15) return nk_misaligned_k;                                                           \
        if (bytes > NUMKONG_SIZE_MAX - (nk_size_t)packed) return nk_unexpected_dimensions_k;                          \
        nk_size_t const row_bytes = nk_cross_padded_values_serial_(depth, depth_width, per_value,                     \
                                                                   sizeof(nk_##raw_type##_t)) *                       \
                                    sizeof(nk_##raw_type##_t);                                                        \
        nk_size_t const scales_stride = nk_cross_scales_stride_serial_(nk_##dtype##_k, depth);                        \
        nk_cross_operand_t b = {packed, NULL, scales_stride, NULL};                                                   \
        if (nk_block_scaled_format_of_dtype(nk_##dtype##_k).block_size) {                                             \
            b.scales = (nk_u8_t const *)packed + sizeof(nk_cross_packed_buffer_header_t) + columns * row_bytes;       \
            b.tensor_scale = (nk_f32_t const *)((nk_u8_t const *)packed +                                             \
                                                offsetof(nk_cross_packed_buffer_header_t, tensor_scale));             \
        }                                                                                                             \
        return nk_cross_launch_##isa##_(nk_cross_small_int4_metal_(nk_##dtype##_k, nk_cap_##isa##_k, rows, columns)   \
                                            ? "nk_dots_" #dtype "_" #isa "_small_kernel_"                             \
                                            : "nk_dots_" #dtype "_" #isa "_kernel_",                                  \
                                        nk_cross_operand_serial_(nk_##dtype##_k, a, a_stride), b,                     \
                                        sizeof(nk_cross_packed_buffer_header_t), c, sizeof(nk_##result_type##_t), 0,  \
                                        rows, columns, depth, depth / per_value * sizeof(nk_##raw_type##_t),          \
                                        nk_block_scaled_format_of_dtype(nk_##dtype##_k).block_size                    \
                                            ? columns * (scales_stride + sizeof(nk_##norm_type##_t))                  \
                                            : 0,                                                                      \
                                        a_stride, row_bytes, c_stride, 0,                                             \
                                        nk_cross_tile_side_metal_(nk_##dtype##_k, nk_cap_##isa##_k, rows, columns),   \
                                        nk_cross_scale_blocks_serial_(nk_##dtype##_k, depth), stream);                \
    }                                                                                                                 \
    NUMKONG_API nk_status_t nk_dots_symmetric_##dtype##_##isa(                                                        \
        nk_cross_##dtype##_operand_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride,       \
        nk_##result_type##_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end,              \
        nk_stream_t stream) {                                                                                         \
        if (!nk_cross_whole_blocks_serial_(nk_##dtype##_k, depth)) return nk_unexpected_dimensions_k;                 \
        rows_end = nk_min_of_two(rows_end, vector_count);                                                             \
        nk_size_t const window_rows = rows_end > rows_begin ? rows_end - rows_begin : 0;                              \
        nk_cross_operand_t const a = nk_cross_operand_serial_(nk_##dtype##_k, vectors, stride);                       \
        return nk_cross_launch_##isa##_(                                                                              \
            nk_cross_small_int4_metal_(nk_##dtype##_k, nk_cap_##isa##_k, window_rows, vector_count)                   \
                ? "nk_dots_" #dtype "_" #isa "_small_kernel_"                                                         \
                : "nk_dots_" #dtype "_" #isa "_kernel_",                                                              \
            a, a, 0, result, sizeof(nk_##result_type##_t), rows_begin, rows_end, vector_count, depth,                 \
            depth / per_value * sizeof(nk_##raw_type##_t), 0, stride, stride, result_stride, 1,                       \
            nk_cross_tile_side_metal_(nk_##dtype##_k, nk_cap_##isa##_k, window_rows, vector_count),                   \
            nk_cross_scale_blocks_serial_(nk_##dtype##_k, depth), stream);                                            \
    }

#pragma endregion Cross Macros

#if NUMKONG_TARGET_METAL
nk_define_cross_metal_(i8, metal, i8, i32, u32, 16, 1)
nk_define_cross_metal_(u8, metal, u8, u32, u32, 16, 1)
nk_define_cross_metal_(i4, metal, i4x2, i32, u32, 32, 2)
nk_define_cross_metal_(u4, metal, u4x2, u32, u32, 32, 2)
nk_define_cross_metal_(f16, metal, f16, f32, f32, 16, 1)
nk_define_cross_metal_(bf16, metal, bf16, f32, f32, 16, 1)
nk_define_cross_metal_(e4m3, metal, e4m3, f32, f32, 16, 1)
nk_define_cross_metal_(e5m2, metal, e5m2, f32, f32, 16, 1)
nk_define_cross_metal_(e3m2, metal, e3m2, f32, f32, 16, 1)
nk_define_cross_metal_(e2m3, metal, e2m3, f32, f32, 16, 1)
nk_define_cross_metal_(e2m1, metal, e2m1x2, f32, f32, 32, 2)
nk_define_cross_metal_(mxfp8e4m3, metal, e4m3, f32, f32, 32, 1)
nk_define_cross_metal_(mxfp8e5m2, metal, e5m2, f32, f32, 32, 1)
nk_define_cross_metal_(mxfp6e2m3, metal, e2m3, f32, f32, 32, 1)
nk_define_cross_metal_(mxfp6e3m2, metal, e3m2, f32, f32, 32, 1)
nk_define_cross_metal_(mxfp4, metal, e2m1x2, f32, f32, 32, 2)
nk_define_cross_metal_(nvfp4, metal, e2m1x2, f32, f32, 32, 2)
#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_DOTS_METAL_H
