/**
 *  @file include/numkong/dots/simt.h
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Batched Dot Products on the SIMT cores of every Apple GPU, launched from C.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/simt.metal, the kernels this embeds
 *  @sa include/numkong/dots/simt.cuh, the CUDA and ROCm sibling
 *
 *  Every signature matches the @c cuda capability's, with an @ref nk_metal_queue_t as the stream.
 *  The kernels travel as their own `.metal` source, embedded here and compiled on the queue at
 *  first use under Metal 3.1, the first with @c bfloat, so every Apple GPU from family 7 runs them.
 *  The pack, its shape reader and the launch core serve @c apple9 and @c apple10 too.
 */
#ifndef NUMKONG_DOTS_SIMT_H
#define NUMKONG_DOTS_SIMT_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h"       // `nk_metal_queue_t`
#include "numkong/dots/serial.h" // `nk_cross_packed_buffer_header_t`, `nk_cross_padded_values_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per queue. */
static char const nk_dots_metal_source_[] = {
#embed "simt.metal" suffix(, 0)
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
    nk_u32_t row_start, row_end, column_count, depth;
    nk_u32_t upper_triangle;
} nk_cross_arguments_metal_t;

/** The launch record of a pack kernel, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t b_stride_bytes, depth_bytes, row_bytes;
    nk_capability_t capability;
    nk_u32_t column_count, depth, depth_padded_values, columns_begin, columns_end;
} nk_cross_pack_arguments_metal_t;

nk_static_assert_(sizeof(nk_cross_arguments_metal_t) == 48, nk_cross_arguments_metal_must_be_48_bytes);
nk_static_assert_(sizeof(nk_cross_pack_arguments_metal_t) == 56, nk_cross_pack_arguments_metal_must_be_56_bytes);

/** Bytes of a packed B: the header, every padded row, and one norm per column. */
NUMKONG_INLINE nk_size_t nk_cross_pack_size_metal_(nk_size_t width, nk_size_t depth, nk_size_t depth_simd_dimensions,
                                                   nk_size_t value_bytes, nk_size_t norm_bytes,
                                                   nk_size_t dimensions_per_value) {
    nk_size_t const values = nk_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value, value_bytes);
    return sizeof(nk_cross_packed_buffer_header_t) + width * values * value_bytes + width * norm_bytes;
}

/** Synchronizes, then reads the width and depth out of a packed B's header, if @p capability
 *  packed it. */
NUMKONG_INLINE nk_status_t nk_cross_packed_shape_metal_(void const *b_packed, nk_size_t *width, nk_size_t *depth,
                                                        nk_capability_t capability, void *stream) {
    nk_status_t const status = nk_metal_synchronize((nk_metal_queue_t *)stream);
    if (status != nk_success_k) return status;
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    if (header->capability != capability) return nk_pack_mismatch_k;
    *width = header->column_count, *depth = header->depth_dimensions;
    return nk_success_k;
}

/** Encodes one pack of columns [begin,end), one simdgroup per column, recording the packing
 *  @p capability. */
NUMKONG_INLINE nk_status_t nk_cross_pack_launch_metal_(char const *kernel, void const *b, nk_size_t width,
                                                       nk_size_t depth, nk_size_t b_stride, void *b_packed,
                                                       nk_size_t columns_begin, nk_size_t columns_end,
                                                       nk_size_t depth_simd_dimensions, nk_size_t value_bytes,
                                                       nk_size_t dimensions_per_value, nk_capability_t capability,
                                                       void *stream) {
    nk_metal_queue_t *queue = (nk_metal_queue_t *)stream;
    if (columns_end > width) columns_end = width;
    if (columns_end <= columns_begin && columns_begin != 0) return nk_success_k;
    nk_size_t b_offset = 0, packed_offset = 0;
    nk_metal_allocation_t const *b_block = nk_metal_resolve_(queue, b, &b_offset);
    nk_metal_allocation_t const *packed_block = nk_metal_resolve_(queue, b_packed, &packed_offset);
    if (!packed_block || (!b_block && columns_end > columns_begin)) return nk_device_memory_mismatch_k;
    void *const pipeline = nk_metal_pipeline_(queue, nk_dots_metal_source_, kernel, NUMKONG_METAL_LANGUAGE_3_1_);
    void *const encoder = pipeline ? nk_metal_encoder_(queue) : NULL;
    if (!encoder) return queue->status;

    nk_cross_pack_arguments_metal_t arguments;
    nk_size_t const values = nk_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value, value_bytes);
    arguments.b_stride_bytes = b_stride, arguments.depth_bytes = depth / dimensions_per_value * value_bytes;
    arguments.row_bytes = values * value_bytes;
    arguments.column_count = (nk_u32_t)width, arguments.depth = (nk_u32_t)depth;
    arguments.depth_padded_values = (nk_u32_t)values;
    arguments.columns_begin = (nk_u32_t)columns_begin, arguments.columns_end = (nk_u32_t)columns_end;
    arguments.capability = capability;

    // An empty window still writes the header when it starts at column zero, so one group runs.
    nk_size_t const simdgroups_per_group = 8;
    nk_size_t const columns = columns_end > columns_begin ? columns_end - columns_begin : 0;
    nk_size_t const groups = columns ? nk_size_divide_round_up_(columns, simdgroups_per_group) : 1;
    nk_metal_bind_(encoder, b_block ? b_block : packed_block, b_block ? b_offset : packed_offset, 0);
    nk_metal_bind_(encoder, packed_block, packed_offset, 1);
    nk_metal_bind_bytes_(encoder, &arguments, sizeof(arguments), 2);
    nk_metal_size_t const grid = {groups, 1, 1}, threads = {simdgroups_per_group * 32, 1, 1};
    nk_metal_dispatch_(queue, pipeline, grid, threads);
    return nk_success_k;
}

/** Validates the contract and encodes C = A × Bᵀ for the rows from @p row_start up to @p row_end,
 *  one threadgroup of @p threads per @b [64,64] tile, running @p kernel from the library @p source
 *  builds under @p language_version. All kernels own tiles of that side and read this one record,
 *  with A and B strides in bytes, so a capability's launcher binds only the first three. */
NUMKONG_INLINE nk_status_t nk_cross_launch_msl_(char const *source, nk_size_t language_version, nk_size_t threads,
                                                char const *kernel, void const *a, void const *b,
                                                nk_size_t b_extra_offset, void *c, nk_size_t result_bytes,
                                                nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                nk_size_t depth, nk_size_t a_stride, nk_size_t b_stride,
                                                nk_size_t c_stride, nk_u32_t upper_triangle, void *stream) {
    nk_metal_queue_t *queue = (nk_metal_queue_t *)stream;
    if ((((nk_size_t)a) | a_stride) & 15 || (((nk_size_t)c) | c_stride) & (result_bytes - 1)) return nk_misaligned_k;
    if (row_end <= row_start || column_count == 0) return nk_success_k;
    nk_size_t a_offset = 0, b_offset = 0, c_offset = 0;
    nk_metal_allocation_t const *a_block = nk_metal_resolve_(queue, a, &a_offset);
    nk_metal_allocation_t const *b_block = nk_metal_resolve_(queue, b, &b_offset);
    nk_metal_allocation_t const *c_block = nk_metal_resolve_(queue, c, &c_offset);
    if (!a_block || !b_block || !c_block) return nk_device_memory_mismatch_k;
    void *const pipeline = nk_metal_pipeline_(queue, source, kernel, language_version);
    void *const encoder = pipeline ? nk_metal_encoder_(queue) : NULL;
    if (!encoder) return queue->status;

    nk_cross_arguments_metal_t arguments;
    arguments.a_stride = a_stride, arguments.b_stride = b_stride, arguments.c_stride = c_stride / result_bytes;
    arguments.row_start = (nk_u32_t)row_start, arguments.row_end = (nk_u32_t)row_end;
    arguments.column_count = (nk_u32_t)column_count, arguments.depth = (nk_u32_t)depth;
    arguments.upper_triangle = upper_triangle;
    nk_metal_bind_(encoder, a_block, a_offset, 0);
    nk_metal_bind_(encoder, b_block, b_offset + b_extra_offset, 1);
    nk_metal_bind_(encoder, c_block, c_offset, 2);
    nk_metal_bind_bytes_(encoder, &arguments, sizeof(arguments), 3);
    nk_metal_size_t const grid = {nk_size_divide_round_up_(column_count, nk_cross_tile_metal_k),
                                  nk_size_divide_round_up_(row_end - row_start, nk_cross_tile_metal_k), 1};
    nk_metal_size_t const group = {threads, 1, 1};
    nk_metal_dispatch_(queue, pipeline, grid, group);
    return nk_success_k;
}

/** @ref nk_cross_launch_msl_ over the @c metal capability's kernels. */
NUMKONG_INLINE nk_status_t nk_cross_launch_metal_(char const *kernel, void const *a, void const *b,
                                                  nk_size_t b_extra_offset, void *c, nk_size_t result_bytes,
                                                  nk_size_t row_start, nk_size_t row_end, nk_size_t column_count,
                                                  nk_size_t depth, nk_size_t a_stride, nk_size_t b_stride,
                                                  nk_size_t c_stride, nk_u32_t upper_triangle, void *stream) {
    return nk_cross_launch_msl_(nk_dots_metal_source_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cross_threads_metal_k, kernel, a,
                                b, b_extra_offset, c, result_bytes, row_start, row_end, column_count, depth, a_stride,
                                b_stride, c_stride, upper_triangle, stream);
}

#pragma endregion Launchers

#pragma region Cross Macros

/** Every entry of one dtype on the Metal capability @p isa_suffix: the @c metal pack kernel with
 *  rows padded to @p depth_simd_dimensions, and one tile kernel serving both the packed and the
 *  symmetric entries, encoded by that capability's launcher. */
#define nk_define_device_cross_msl_(input_type_name, isa_suffix, input_value_type, result_value_type, norm_value_type, \
                                    depth_simd_dimensions, dimensions_per_value)                                       \
    NUMKONG_API nk_status_t nk_dots_pack_size_##input_type_name##_##isa_suffix(nk_size_t width, nk_size_t depth,       \
                                                                               nk_size_t *bytes) {                     \
        *bytes = nk_cross_pack_size_metal_(width, depth, depth_simd_dimensions, sizeof(nk_##input_value_type##_t),     \
                                           sizeof(nk_##norm_value_type##_t), dimensions_per_value);                    \
        return nk_success_k;                                                                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_dots_packed_shape_##input_type_name##_##isa_suffix(                                     \
        void const *b_packed, nk_size_t *width, nk_size_t *depth, void *stream) {                                      \
        return nk_cross_packed_shape_metal_(b_packed, width, depth, nk_cap_##isa_suffix##_k, stream);                  \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_dots_pack_##input_type_name##_##isa_suffix(                                             \
        nk_##input_value_type##_t const *b, void const *b_scales, nk_size_t width, nk_size_t depth,                    \
        nk_size_t b_stride, nk_size_t b_scales_stride, void *b_packed, nk_size_t columns_begin, nk_size_t columns_end, \
        void *stream) {                                                                                                \
        return nk_cross_pack_launch_metal_("nk_dots_pack_" #input_type_name "_metal_kernel_", b, width, depth,         \
                                           b_stride, b_packed, columns_begin, columns_end, depth_simd_dimensions,      \
                                           sizeof(nk_##input_value_type##_t), dimensions_per_value,                    \
                                           nk_cap_##isa_suffix##_k, stream);                                           \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_dots_packed_##input_type_name##_##isa_suffix(                                           \
        nk_##input_value_type##_t const *a, void const *a_scales, void const *b_packed, nk_##result_value_type##_t *c, \
        nk_size_t height, nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t a_scales_stride,             \
        nk_size_t c_stride, void *stream) {                                                                            \
        nk_size_t const row_bytes = nk_cross_padded_values_(depth, depth_simd_dimensions, dimensions_per_value,        \
                                                            sizeof(nk_##input_value_type##_t)) *                       \
                                    sizeof(nk_##input_value_type##_t);                                                 \
        return nk_cross_launch_##isa_suffix##_("nk_dots_" #input_type_name "_" #isa_suffix "_kernel_", a, b_packed,    \
                                               sizeof(nk_cross_packed_buffer_header_t), c,                             \
                                               sizeof(nk_##result_value_type##_t), 0, height, width, depth, a_stride,  \
                                               row_bytes, c_stride, 0, stream);                                        \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_dots_symmetric_##input_type_name##_##isa_suffix(                                        \
        nk_##input_value_type##_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth, \
        nk_size_t stride, nk_size_t scales_stride, nk_##result_value_type##_t *result, nk_size_t result_stride,        \
        nk_size_t row_start, nk_size_t row_count, void *stream) {                                                      \
        nk_size_t const row_end = row_start + row_count < vectors_count ? row_start + row_count : vectors_count;       \
        return nk_cross_launch_##isa_suffix##_("nk_dots_" #input_type_name "_" #isa_suffix "_kernel_", vectors,        \
                                               vectors, 0, result, sizeof(nk_##result_value_type##_t), row_start,      \
                                               row_end, vectors_count, depth, stride, stride, result_stride, 1,        \
                                               stream);                                                                \
    }

#pragma endregion Cross Macros

#if NUMKONG_TARGET_METAL
nk_define_device_cross_msl_(i8, metal, i8, i32, u32, 16, 1)
nk_define_device_cross_msl_(u8, metal, u8, u32, u32, 16, 1)
nk_define_device_cross_msl_(i4, metal, i4x2, i32, u32, 32, 2)
nk_define_device_cross_msl_(u4, metal, u4x2, u32, u32, 32, 2)
nk_define_device_cross_msl_(f16, metal, f16, f32, f32, 16, 1)
nk_define_device_cross_msl_(bf16, metal, bf16, f32, f32, 16, 1)
nk_define_device_cross_msl_(e4m3, metal, e4m3, f32, f32, 16, 1)
nk_define_device_cross_msl_(e5m2, metal, e5m2, f32, f32, 16, 1)
nk_define_device_cross_msl_(e3m2, metal, e3m2, f32, f32, 16, 1)
nk_define_device_cross_msl_(e2m3, metal, e2m3, f32, f32, 16, 1)
nk_define_device_cross_msl_(e2m1, metal, e2m1x2, f32, f32, 32, 2)
#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_DOTS_SIMT_H
