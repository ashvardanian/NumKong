/**
 *  @file include/numkong/shapes/metal.h
 *  @author Ash Vardanian
 *  @date October 24, 2026
 *  @brief Chamfer and Hausdorff distances between F16 point clouds on the SIMT cores of every
 *      Apple GPU, launched from C.
 *
 *  @sa include/numkong/shapes.h
 *  @sa include/numkong/shapes/metal.metal, the kernels this embeds
 *  @sa include/numkong/shapes/blackwell.cuh, the CUDA sibling of the many-to-many kernels
 *
 *  Every signature matches the @c blackwell capability's, with an @c id<MTLCommandQueue> as the
 *  stream. The pack tables, the packs, the distances and the rotations are unified memory from
 *  @ref nk_memory_allocate_unified_metal, and the packs of one launch may span at most 64 of its
 *  allocations. The launcher reads every pack's header on the host to resolve its allocation,
 *  and the kernels translate the host addresses of the tables through those allocations.
 */
#ifndef NUMKONG_SHAPES_METAL_H
#define NUMKONG_SHAPES_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h"         // `nk_metal_call_t`
#include "numkong/shapes/serial.h" // `nk_shapes_packed_header_f32_t`, `nk_shapes_alignment_k_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_shapes_source_metal_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/shapes/metal.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Threads of every shapes threadgroup: four simdgroups. */
enum { nk_shapes_threads_metal_k = 128, nk_shapes_blocks_metal_k = 64 };

/** One allocation the packs lie in, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t host_begin, host_end, device_begin;
} nk_shapes_block_metal_t;

/** The launch record of every shapes kernel, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t rows, columns, results_stride;
    nk_u32_t blocks, rotations;
    nk_shapes_block_metal_t block_list[nk_shapes_blocks_metal_k];
} nk_shapes_arguments_metal_t;

nk_static_assert_(sizeof(nk_shapes_arguments_metal_t) == 32 + 24 * nk_shapes_blocks_metal_k,
                  nk_shapes_arguments_metal_must_match_the_kernels);

/**
 *  @brief Resolves the @p count packs of @p table into the allocations of @p arguments, adding each
 *      new one to the call's resources once.
 *  @param[in] points The points every pack must hold, or @c NUMKONG_SIZE_MAX for any.
 *  @return @c nk_misaligned_k for a pack off an @c nk_default_alignment_k boundary,
 *      @c nk_unexpected_dimensions_k for a pack of other @p points or a 65th allocation, or
 *      @c nk_device_memory_mismatch_k for a pack outside unified memory.
 */
NUMKONG_INLINE nk_status_t nk_shapes_resolve_packs_metal_(nk_metal_call_t *call, void const *const *table,
                                                          nk_size_t count, nk_size_t points,
                                                          nk_shapes_arguments_metal_t *arguments) {
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    for (nk_size_t index = 0; index != count; ++index) {
        nk_size_t const address = (nk_size_t)table[index];
        if (address & (nk_default_alignment_k - 1)) return nk_misaligned_k;
        nk_size_t const pack_points = ((nk_shapes_packed_header_f32_t const *)table[index])->points;
        if (points != NUMKONG_SIZE_MAX && pack_points != points) return nk_unexpected_dimensions_k;
        nk_size_t const bytes = header_bytes + 3 * nk_size_round_up_to_multiple_(pack_points * sizeof(nk_f16_t),
                                                                                 nk_shapes_alignment_k_);
        nk_u32_t block = 0;
        while (block != arguments->blocks &&
               (address - arguments->block_list[block].host_begin >
                    arguments->block_list[block].host_end - arguments->block_list[block].host_begin ||
                bytes > arguments->block_list[block].host_end - address))
            ++block;
        if (block != arguments->blocks) continue;
        if (block == nk_shapes_blocks_metal_k) return nk_unexpected_dimensions_k;
        nk_size_t offset = 0;
        void *const buffer = nk_resolve_metal_(call, table[index], bytes, &offset);
        if (!buffer) return nk_device_memory_mismatch_k;
        nk_shapes_block_metal_t *const resolved = &arguments->block_list[arguments->blocks++];
        resolved->host_begin = address - offset;
        resolved->host_end = resolved->host_begin + nk_count_metal_(buffer, "length");
        resolved->device_begin = nk_count_metal_(buffer, "gpuAddress");
    }
    return nk_success_k;
}

/**
 *  @brief Validates the contract and encodes @p kernel over every pair of @p rows row packs and
 *      @p columns column packs, @p groups threadgroups of 128 threads in all.
 *  @param[in] points The points every pack must hold, or @c NUMKONG_SIZE_MAX for any.
 */
NUMKONG_INLINE nk_status_t nk_shapes_launch_metal_(char const *source, nk_size_t language_version, char const *kernel,
                                                   nk_size_t groups, nk_size_t points, void const *const *a_packed,
                                                   nk_size_t rows, void const *const *b_packed, nk_size_t columns,
                                                   nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                   nk_stream_t stream) {
    if (!rows || !columns) return nk_success_k;
    if (((nk_size_t)results | results_stride | (nk_size_t)rotations) & (sizeof(nk_f32_t) - 1)) return nk_misaligned_k;
    nk_size_t row_bytes, results_bytes, pairs, rotations_bytes;
    if (!nk_size_mul_checked_(columns, sizeof(nk_f32_t), &row_bytes) ||
        !nk_size_span_checked_(rows, results_stride, row_bytes, &results_bytes) ||
        !nk_size_mul_checked_(rows, columns, &pairs) ||
        !nk_size_mul_checked_(pairs, 9 * sizeof(nk_f32_t), &rotations_bytes))
        return nk_unexpected_dimensions_k;

    nk_metal_call_t call;
    nk_status_t status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_shapes_arguments_metal_t arguments;
    arguments.rows = rows, arguments.columns = columns, arguments.results_stride = results_stride;
    arguments.blocks = 0, arguments.rotations = rotations != NULL;
    status = nk_shapes_resolve_packs_metal_(&call, a_packed, rows, points, &arguments);
    if (status == nk_success_k) status = nk_shapes_resolve_packs_metal_(&call, b_packed, columns, points, &arguments);
    if (status != nk_success_k) return nk_abort_metal_(&call, status);
    nk_size_t a_offset = 0, b_offset = 0, results_offset = 0, rotations_offset = 0;
    void *const a_buffer = nk_resolve_metal_(&call, a_packed, rows * sizeof(void const *), &a_offset);
    void *const b_buffer = nk_resolve_metal_(&call, b_packed, columns * sizeof(void const *), &b_offset);
    void *const results_buffer = nk_resolve_metal_(&call, results, results_bytes, &results_offset);
    void *const rotations_buffer = rotations ? nk_resolve_metal_(&call, rotations, rotations_bytes, &rotations_offset)
                                             : results_buffer;
    if (!a_buffer || !b_buffer || !results_buffer || !rotations_buffer)
        return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, source, kernel, language_version);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    nk_bind_metal_(call.encoder, a_buffer, a_offset, 0);
    nk_bind_metal_(call.encoder, b_buffer, b_offset, 1);
    nk_bind_metal_(call.encoder, results_buffer, results_offset, 2);
    nk_bind_metal_(call.encoder, rotations_buffer, rotations ? rotations_offset : results_offset, 3);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);
    // Threadgroups fill rows of 65535, which every Apple GPU's grid takes in each dimension
    nk_size_t const width = groups < 65535 ? groups : 65535;
    nk_metal_size_t const grid = {width, nk_size_divide_round_up_(groups, width), 1};
    nk_metal_size_t const threads = {nk_shapes_threads_metal_k, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

#if NUMKONG_TARGET_METAL

NUMKONG_API nk_status_t nk_chamfers_packed_f16_metal(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                     nk_size_t results_stride, nk_f32_t *rotations,
                                                     nk_stream_t stream) {
    return nk_shapes_launch_metal_(nk_shapes_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_,
                                   "nk_chamfers_packed_f16_metal_kernel_", nk_size_divide_round_up_(rows * columns, 4),
                                   NUMKONG_SIZE_MAX, a_packed, rows, b_packed, columns, results, results_stride,
                                   rotations, stream);
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_metal(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_stream_t stream) {
    return nk_shapes_launch_metal_(nk_shapes_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_,
                                   "nk_hausdorffs_packed_f16_metal_kernel_",
                                   nk_size_divide_round_up_(rows * columns, 4), NUMKONG_SIZE_MAX, a_packed, rows,
                                   b_packed, columns, results, results_stride, rotations, stream);
}

#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_SHAPES_METAL_H
