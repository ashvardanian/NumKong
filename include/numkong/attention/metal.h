/**
 *  @file include/numkong/attention/metal.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Ragged attention and rotary position embeddings on the SIMT cores of every Apple GPU,
 *      launched from C.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/metal.metal, the kernels this embeds
 *  @sa include/numkong/attention/cuda.cuh, the CUDA sibling
 *
 *  Every signature matches the @c cuda capability's, with an @c id<MTLCommandQueue> as the stream,
 *  and block-scaled entries take their codes, block scales and NVFP4 tensor scale by reference, as
 *  the serial ones do. The pack, its shape reader and the launch core serve @c apple9 and
 *  @c apple10 too. The kernels travel as their own `.metal` source, compiled on the device at first
 *  use under Metal 3.1, the first with @c bfloat. Only the shape reader reads device memory on the
 *  host, after waiting on the stream: the other entries trust the pack and the query offsets, and
 *  the kernels read the tensor scales. Attention checks that the @c query_token_count rows of the
 *  queries, their scales and the outputs lie in their buffers, and its kernels stay below that row
 *  and inside the bytes the pack and offsets hold past their pointers. Gradients also keep the key
 *  and value gradient rows inside the bytes their buffers hold.
 *
 *  The @c metal baseline gives every query row a SIMD-group, while @c apple9 and @c apple10 attend
 *  32-row tiles against 64-key panels, on SIMD-group matrices and on the @c matmul2d tensor
 *  operations. Calls of at most four query rows against 512 keys or more split each row's keys into
 *  16 partitions, whose maxima, sums and weighted values a merge kernel combines in a fixed order.
 *  The baseline's gradients take a SIMD-group per key for dK and dV, then one per query row for dQ,
 *  while @c apple9 and @c apple10 take 32-key tiles, then 32-row tiles, on their matrices.
 */
#ifndef NUMKONG_ATTENTION_METAL_H
#define NUMKONG_ATTENTION_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h"            // `nk_metal_call_t`
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_bound_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_attention_source_metal_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/attention/metal.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Launchers

/** The launch record of a RoPE kernel, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t rows, heads, depth, x_stride, y_stride;
} nk_attention_rope_arguments_metal_t;

/** The launch record of the pack kernels, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t key_value_head_count, depth, head_bytes, segments, key_stride, value_stride, tasks_begin, tasks_end;
    nk_u64_t key_bytes, value_bytes, packed_bytes, lengths_bytes, capability;
    nk_u64_t code_bytes, key_scales_stride, value_scales_stride, key_scales_bytes, value_scales_bytes;
    nk_u32_t key_tensor_present, value_tensor_present;
} nk_attention_pack_arguments_metal_t;

/** The launch record of the attention kernels, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t heads, kv_heads, depth, query_tokens, query_stride, output_stride, keys_before, keys_after;
    nk_u64_t tasks_begin, tasks_end, packed_bytes, offsets_bytes, log_sum_exp_bytes, capability;
    nk_f32_t scale2;
    nk_u32_t query_tensor_present;
    nk_u64_t query_scales_stride;
} nk_attention_arguments_metal_t;

/** The launch record of the backward kernels, laid out as the kernels' record of the same name. */
typedef struct {
    nk_attention_arguments_metal_t attention;
    nk_u64_t query_gradient_stride, key_value_gradient_stride, key_value_gradient_bytes;
    nk_f32_t scale;
} nk_attention_backward_arguments_metal_t;

nk_static_assert_(sizeof(nk_attention_rope_arguments_metal_t) == 40,
                  nk_attention_rope_arguments_metal_must_be_40_bytes);
nk_static_assert_(sizeof(nk_attention_pack_arguments_metal_t) == 152,
                  nk_attention_pack_arguments_metal_must_be_152_bytes);
nk_static_assert_(sizeof(nk_attention_arguments_metal_t) == 128, nk_attention_arguments_metal_must_be_128_bytes);
nk_static_assert_(sizeof(nk_attention_backward_arguments_metal_t) == 160,
                  nk_attention_backward_arguments_metal_must_be_160_bytes);

/** Encodes @p groups threadgroups of @p threads each and a barrier after them, without committing,
 *  for calls that chain kernels. */
NUMKONG_INLINE void nk_attention_encode_metal_(nk_metal_call_t *call, nk_metal_size_t groups, nk_metal_size_t threads) {
    ((void (*)(void *, SEL, nk_metal_size_t, nk_metal_size_t))objc_msgSend)(
        call->encoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
    nk_size_t const buffers_scope = 1; // `MTLBarrierScopeBuffers`
    ((void (*)(void *, SEL, nk_size_t))objc_msgSend)(call->encoder, sel_registerName("memoryBarrierWithScope:"),
                                                     buffers_scope);
}

/** Validates the contract and encodes @p kernel with a simdgroup per head of every row, eight per
 *  threadgroup, at most 2²⁰ threadgroups of them. */
NUMKONG_INLINE nk_status_t nk_attention_rope_launch_metal_(char const *kernel, nk_size_t value_bytes, void const *x,
                                                           nk_f32_t const *cos, nk_f32_t const *sin, void *y,
                                                           nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                           nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    if (depth % 2) return nk_unexpected_dimensions_k;
    if (((((nk_size_t)x) | x_stride | ((nk_size_t)y) | y_stride) & (value_bytes - 1)) ||
        ((((nk_size_t)cos) | ((nk_size_t)sin)) & 3))
        return nk_misaligned_k;
    if (!rows || !head_count || !depth) return nk_success_k;
    nk_size_t row_bytes, heads, table_bytes, x_bytes, y_bytes;
    if (depth > NUMKONG_U32_MAX || !nk_size_mul_checked_(head_count, depth, &row_bytes) ||
        !nk_size_mul_checked_(row_bytes, value_bytes, &row_bytes) || !nk_size_mul_checked_(rows, head_count, &heads) ||
        !nk_size_mul_checked_(rows, depth / 2 * sizeof(nk_f32_t), &table_bytes) ||
        !nk_size_span_checked_(rows, x_stride, row_bytes, &x_bytes) ||
        !nk_size_span_checked_(rows, y_stride, row_bytes, &y_bytes))
        return nk_unexpected_dimensions_k;

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t offsets[4] = {0};
    void *const buffers[4] = {
        nk_resolve_metal_(&call, x, x_bytes, offsets), nk_resolve_metal_(&call, cos, table_bytes, offsets + 1),
        nk_resolve_metal_(&call, sin, table_bytes, offsets + 2), nk_resolve_metal_(&call, y, y_bytes, offsets + 3)};
    for (nk_size_t index = 0; index != 4; ++index)
        if (!buffers[index]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, nk_attention_source_metal_, kernel,
                                              NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    for (nk_size_t index = 0; index != 4; ++index) nk_bind_metal_(call.encoder, buffers[index], offsets[index], index);
    nk_attention_rope_arguments_metal_t const arguments = {rows, head_count, depth, x_stride, y_stride};
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);
    nk_size_t const groups = nk_size_divide_round_up_(heads, 8), groups_limit = (nk_size_t)1 << 20;
    nk_metal_size_t const grid = {groups < groups_limit ? groups : groups_limit, 1, 1}, threads = {256, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/** Bounds the pack of @p token_count keys of @p element_bytes per element in @p segment_count
 *  segments, however they split: the header, the directory and both planes of every head. */
NUMKONG_INLINE nk_status_t nk_attention_pack_size_metal_(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t element_bytes, nk_size_t *bytes) {
    *bytes = nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                             depth * element_bytes);
    return nk_success_k;
}

/** Synchronizes, then reads the shape out of a pack's header, if @p capability packed it. */
NUMKONG_INLINE nk_status_t nk_attention_packed_shape_metal_(void const *packed, nk_size_t *key_value_head_count,
                                                            nk_size_t *depth, nk_size_t *segments,
                                                            nk_capability_t capability, nk_stream_t stream) {
    if ((nk_size_t)packed & 15) return nk_misaligned_k;
    nk_status_t const status = nk_stream_synchronize_metal_(stream);
    if (status != nk_success_k) return status;
    nk_metal_call_t call;
    nk_status_t const entered = nk_enter_metal_(stream, &call);
    if (entered != nk_success_k) return entered;
    nk_size_t offset;
    if (!nk_resolve_metal_(&call, packed, sizeof(nk_attention_packed_header_t), &offset))
        return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)packed;
    if (header->capability != capability) return nk_abort_metal_(&call, nk_pack_mismatch_k);
    *key_value_head_count = header->key_value_head_count, *depth = header->depth, *segments = header->segments;
    return nk_abort_metal_(&call, nk_success_k);
}

/** Encodes @p directory_kernel, recording the packing @p capability, for a window starting at task
 *  0, then @p pack_kernel over the window, a threadgroup per task. Rows of @p key_value_head_count
 *  heads hold @p code_bytes of codes per head and, for block-scaled operands, @p scale_blocks block
 *  scales per head, while the planes hold @p element_bytes per element. */
NUMKONG_INLINE nk_status_t nk_attention_pack_launch_metal_(
    nk_capability_t capability, char const *directory_kernel, char const *pack_kernel, nk_size_t element_bytes,
    nk_size_t code_bytes, nk_size_t scale_blocks, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count, nk_cross_operand_t keys,
    nk_size_t key_stride, nk_cross_operand_t values, nk_size_t value_stride, void *packed, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    if (((nk_size_t)packed & 15) ||
        ((((nk_size_t)keys.elements) | key_stride | ((nk_size_t)values.elements) | value_stride) &
         (scale_blocks ? 0 : element_bytes - 1)) ||
        ((((nk_size_t)key_offsets) | ((nk_size_t)key_lengths)) & 3))
        return nk_misaligned_k;
    if (key_value_head_count > NUMKONG_U32_MAX || depth > NUMKONG_U32_MAX || segment_count > NUMKONG_U32_MAX)
        return nk_unexpected_dimensions_k;
    nk_size_t const head_bytes = depth * element_bytes, row_bytes = key_value_head_count * code_bytes;
    nk_size_t const scales_row_bytes = key_value_head_count * scale_blocks;
    nk_size_t const tasks = segment_count * key_value_head_count, end = tasks_end < tasks ? tasks_end : tasks;
    int const payload = tasks_begin < end && row_bytes != 0;
    if (tasks_begin != 0 && !payload) return nk_success_k;
    if (payload && scale_blocks && (!keys.scales || !values.scales)) return nk_device_memory_mismatch_k;
    nk_size_t const directory_bytes = sizeof(nk_attention_packed_header_t) +
                                      nk_attention_pack_directory_size_serial_(segment_count);

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t offsets[10] = {0};
    void *buffers[10];
    buffers[4] = nk_resolve_metal_(&call, packed, directory_bytes, offsets + 4);
    buffers[2] = nk_resolve_metal_(&call, key_offsets, (segment_count + 1) * sizeof(nk_u32_t), offsets + 2);
    if (!buffers[4] || !buffers[2]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    int const lengths = key_lengths && segment_count, scales = payload && scale_blocks;
    buffers[3] = lengths ? nk_resolve_metal_(&call, key_lengths, segment_count * sizeof(nk_u32_t), offsets + 3)
                         : buffers[2];
    buffers[0] = payload ? nk_resolve_metal_(&call, keys.elements, row_bytes, offsets) : buffers[4];
    buffers[1] = payload ? nk_resolve_metal_(&call, values.elements, row_bytes, offsets + 1) : buffers[4];
    buffers[6] = scales ? nk_resolve_metal_(&call, keys.scales, scales_row_bytes, offsets + 6) : buffers[4];
    buffers[7] = scales ? nk_resolve_metal_(&call, values.scales, scales_row_bytes, offsets + 7) : buffers[4];
    buffers[8] = keys.tensor_scale ? nk_resolve_metal_(&call, keys.tensor_scale, sizeof(nk_f32_t), offsets + 8)
                                   : buffers[4];
    buffers[9] = values.tensor_scale ? nk_resolve_metal_(&call, values.tensor_scale, sizeof(nk_f32_t), offsets + 9)
                                     : buffers[4];
    for (nk_size_t index = 0; index != 10; ++index)
        if (index != 5 && !buffers[index]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    if (!lengths) offsets[3] = offsets[2];
    if (!payload) offsets[0] = offsets[1] = offsets[4];
    if (!scales) offsets[6] = offsets[7] = offsets[4];
    if (!keys.tensor_scale) offsets[8] = offsets[4];
    if (!values.tensor_scale) offsets[9] = offsets[4];

    nk_attention_pack_arguments_metal_t arguments;
    arguments.key_value_head_count = key_value_head_count, arguments.depth = depth, arguments.head_bytes = head_bytes;
    arguments.segments = segment_count, arguments.key_stride = key_stride, arguments.value_stride = value_stride;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = end;
    arguments.key_bytes = payload ? nk_count_metal_(buffers[0], "length") - offsets[0] : 0;
    arguments.value_bytes = payload ? nk_count_metal_(buffers[1], "length") - offsets[1] : 0;
    arguments.packed_bytes = nk_count_metal_(buffers[4], "length") - offsets[4];
    arguments.lengths_bytes = lengths ? nk_count_metal_(buffers[3], "length") - offsets[3] : 0;
    arguments.capability = capability, arguments.code_bytes = code_bytes;
    arguments.key_scales_stride = keys.scales_stride, arguments.value_scales_stride = values.scales_stride;
    arguments.key_scales_bytes = scales ? nk_count_metal_(buffers[6], "length") - offsets[6] : 0;
    arguments.value_scales_bytes = scales ? nk_count_metal_(buffers[7], "length") - offsets[7] : 0;
    arguments.key_tensor_present = keys.tensor_scale != NUMKONG_NULL;
    arguments.value_tensor_present = values.tensor_scale != NUMKONG_NULL;
    for (nk_size_t index = 0; index != 10; ++index)
        if (index != 5) nk_bind_metal_(call.encoder, buffers[index], offsets[index], index);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 5);

    if (tasks_begin == 0) {
        void *const pipeline = nk_pipeline_metal_(call.context, nk_attention_source_metal_, directory_kernel,
                                                  NUMKONG_METAL_LANGUAGE_3_1_);
        if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
        nk_encoder_metal_(&call, pipeline);
        nk_metal_size_t const grid = {1, 1, 1}, threads = {32, 1, 1};
        if (!payload) return nk_dispatch_metal_(&call, grid, threads);
        nk_attention_encode_metal_(&call, grid, threads);
    }
    void *const pipeline = nk_pipeline_metal_(call.context, nk_attention_source_metal_, pack_kernel,
                                              NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);
    nk_size_t const groups = end - tasks_begin, groups_limit = 65535;
    nk_metal_size_t const grid = {groups < groups_limit ? groups : groups_limit, 1, 1}, threads = {256, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/**
 *  @brief Validates the contract and encodes the attention of one dtype: @p kernel over the window,
 *      or for calls of up to four query rows against long keys, the split passes and their merge.
 *
 *  The split path cuts each row's keys into 16 partitions: @p split_kernel writes each partition's
 *  maximum, sum and weighted values, then @p weights_kernel, when not null, those of U8 weights
 *  once every maximum is in, and @p merge_kernel combines them. Query rows hold @p code_bytes of
 *  codes per head and, for block-scaled queries, @p scale_blocks block scales per head, while the
 *  planes hold @p element_bytes per element.
 */
NUMKONG_INLINE nk_status_t nk_attention_launch_metal_(
    char const *source, nk_size_t language_version, nk_capability_t capability, char const *kernel,
    char const *split_kernel, char const *weights_kernel, char const *merge_kernel, nk_size_t element_bytes,
    nk_size_t code_bytes, nk_size_t scale_blocks, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *query_offsets, nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
    nk_size_t keys_after, nk_cross_operand_t queries, nk_size_t query_stride, void const *packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (((nk_size_t)packed & 15) ||
        ((((nk_size_t)queries.elements) | query_stride) & (scale_blocks ? 0 : element_bytes - 1)) ||
        ((((nk_size_t)output) | output_stride | ((nk_size_t)log_sum_exp) | ((nk_size_t)query_offsets)) & 3))
        return nk_misaligned_k;
    nk_size_t tasks;
    if (!key_value_head_count || head_count % key_value_head_count || head_count > NUMKONG_U32_MAX ||
        depth > NUMKONG_U32_MAX || !nk_size_mul_checked_(query_token_count, head_count, &tasks))
        return nk_unexpected_dimensions_k;
    if (tasks_end > tasks) tasks_end = tasks;
    if (tasks_begin >= tasks_end || !depth) return nk_success_k;
    if (scale_blocks && !queries.scales) return nk_device_memory_mismatch_k;
    nk_size_t query_row_bytes, output_row_bytes, query_bytes, output_bytes, scales_bytes, log_sum_exp_bytes = 0;
    if (!nk_size_mul_checked_(head_count, code_bytes, &query_row_bytes) ||
        !nk_size_mul_checked_(head_count * depth, sizeof(nk_f32_t), &output_row_bytes) ||
        !nk_size_span_checked_(query_token_count, query_stride, query_row_bytes, &query_bytes) ||
        !nk_size_span_checked_(query_token_count, output_stride, output_row_bytes, &output_bytes) ||
        !nk_size_span_checked_(scale_blocks ? query_token_count : 0, queries.scales_stride, head_count * scale_blocks,
                               &scales_bytes) ||
        (log_sum_exp && !nk_size_mul_checked_(tasks, sizeof(nk_f32_t), &log_sum_exp_bytes)))
        return nk_unexpected_dimensions_k;
    nk_size_t const key_value_bytes = output_row_bytes / sizeof(nk_f32_t) / head_count * key_value_head_count *
                                      element_bytes * 2;

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t offsets[9] = {0};
    void *buffers[9];
    buffers[0] = nk_resolve_metal_(&call, queries.elements, query_bytes, offsets);
    buffers[1] = nk_resolve_metal_(&call, packed, sizeof(nk_attention_packed_header_t), offsets + 1);
    buffers[2] = nk_resolve_metal_(&call, output, output_bytes, offsets + 2);
    buffers[3] = nk_resolve_metal_(&call, query_offsets, 2 * sizeof(nk_u32_t), offsets + 3);
    buffers[5] = log_sum_exp ? nk_resolve_metal_(&call, log_sum_exp, log_sum_exp_bytes, offsets + 5) : buffers[1];
    buffers[7] = scales_bytes ? nk_resolve_metal_(&call, queries.scales, scales_bytes, offsets + 7) : buffers[1];
    buffers[8] = queries.tensor_scale ? nk_resolve_metal_(&call, queries.tensor_scale, sizeof(nk_f32_t), offsets + 8)
                                      : buffers[1];
    if (!log_sum_exp) offsets[5] = offsets[1];
    if (!scales_bytes) offsets[7] = offsets[1];
    if (!queries.tensor_scale) offsets[8] = offsets[1];
    for (nk_size_t index = 0; index != 9; ++index)
        if (index != 4 && index != 6 && !buffers[index]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    nk_size_t const packed_bytes = nk_count_metal_(buffers[1], "length") - offsets[1];
    nk_size_t const offsets_bytes = nk_count_metal_(buffers[3], "length") - offsets[3];

    nk_attention_arguments_metal_t arguments;
    arguments.heads = head_count, arguments.kv_heads = key_value_head_count, arguments.depth = depth;
    arguments.query_tokens = query_token_count, arguments.query_stride = query_stride;
    arguments.output_stride = output_stride, arguments.keys_before = keys_before, arguments.keys_after = keys_after;
    arguments.tasks_begin = tasks_begin, arguments.tasks_end = tasks_end;
    arguments.packed_bytes = packed_bytes, arguments.offsets_bytes = offsets_bytes;
    arguments.log_sum_exp_bytes = log_sum_exp_bytes, arguments.capability = capability;
    arguments.scale2 = scale * NUMKONG_F32_LOG2E_;
    arguments.query_tensor_present = queries.tensor_scale != NUMKONG_NULL;
    arguments.query_scales_stride = queries.scales_stride;
    for (nk_size_t index = 0; index != 9; ++index)
        if (index != 4 && index != 6) nk_bind_metal_(call.encoder, buffers[index], offsets[index], index);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);

    // Threadgroups walk segment × head pairs along x and the window's rows along y
    nk_size_t const rows = nk_size_divide_round_up_(tasks_end, head_count) - tasks_begin / head_count;
    nk_size_t pairs;
    if (!nk_size_mul_checked_(offsets_bytes / sizeof(nk_u32_t) - 1, head_count, &pairs)) pairs = NUMKONG_SIZE_MAX;
    nk_size_t split_bytes;
    if (query_token_count <= 4 && depth <= 256 && keys_before >= 512 && packed_bytes / key_value_bytes >= 512 &&
        nk_size_mul_checked_(tasks, 16 * (depth + 2) * sizeof(nk_f32_t), &split_bytes)) {
        void *const split_pipeline = nk_pipeline_metal_(call.context, source, split_kernel, language_version);
        void *const weights_pipeline = weights_kernel
                                           ? nk_pipeline_metal_(call.context, source, weights_kernel, language_version)
                                           : split_pipeline;
        void *const merge_pipeline = nk_pipeline_metal_(call.context, source, merge_kernel, language_version);
        if (!split_pipeline || !weights_pipeline || !merge_pipeline)
            return nk_abort_metal_(&call, nk_device_code_mismatch_k);
        nk_size_t const private_storage = 32; // `MTLResourceStorageModePrivate`
        void *const split = ((void *(*)(void *, SEL, nk_size_t, nk_size_t))objc_msgSend)(
            call.context->device, sel_registerName("newBufferWithLength:options:"), split_bytes, private_storage);
        if (!split) return nk_abort_metal_(&call, nk_bad_alloc_k);
        nk_bind_metal_(call.encoder, split, 0, 6);
        nk_do_metal_(split, "release");
        nk_size_t const groups = pairs < 65535 / rows ? pairs : 65535 / rows;
        nk_metal_size_t const grid = {groups, rows, 16}, threads = {128, 1, 1};
        nk_encoder_metal_(&call, split_pipeline);
        nk_attention_encode_metal_(&call, grid, threads);
        if (weights_kernel) {
            nk_encoder_metal_(&call, weights_pipeline);
            nk_attention_encode_metal_(&call, grid, threads);
        }
        nk_encoder_metal_(&call, merge_pipeline);
        nk_metal_size_t const merge_grid = {groups, rows, 1}, merge_threads = {32, 1, 1};
        return nk_dispatch_metal_(&call, merge_grid, merge_threads);
    }

    void *const pipeline = nk_pipeline_metal_(call.context, source, kernel, language_version);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);
    nk_size_t const row_groups = nk_min_of_two(32, nk_size_divide_round_up_(rows, 32));
    nk_size_t const groups = pairs < 65535 / row_groups ? pairs : 65535 / row_groups;
    nk_metal_size_t const grid = {groups, row_groups, 1}, threads = {128, 1, 1};
    return nk_dispatch_metal_(&call, grid, threads);
}

/**
 *  @brief Validates the contract and encodes the gradients of one dtype: @p keys_kernel for the key
 *      and value gradients, then @p queries_kernel for the query ones, each threadgroup of
 *      @p threads taking @p tile keys or folded query rows at a time.
 *
 *  Key and value gradient rows lie wherever the pack's key offsets put them, so the kernels keep
 *  them inside the bytes both buffers hold past their pointers.
 */
NUMKONG_INLINE nk_status_t nk_attention_backward_launch_metal_(
    char const *source, nk_size_t language_version, nk_capability_t capability, char const *keys_kernel,
    char const *queries_kernel, nk_size_t tile, nk_size_t threads, nk_size_t element_bytes, nk_size_t head_count,
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_token_count,
    nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, void const *queries, nk_size_t query_stride,
    void const *packed, nk_f32_t const *output, nk_f32_t const *output_gradient, nk_size_t output_stride,
    nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride, nk_f32_t *key_gradient,
    nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin, nk_size_t tasks_end,
    nk_stream_t stream) {
    if (((nk_size_t)packed & 15) || ((((nk_size_t)queries) | query_stride) & (element_bytes - 1)) ||
        ((((nk_size_t)output) | ((nk_size_t)output_gradient) | output_stride | ((nk_size_t)log_sum_exp) |
          ((nk_size_t)query_gradient) | query_gradient_stride | ((nk_size_t)key_gradient) |
          ((nk_size_t)value_gradient) | key_value_gradient_stride | ((nk_size_t)query_offsets)) &
         3))
        return nk_misaligned_k;
    nk_size_t rows;
    if (!key_value_head_count || head_count % key_value_head_count || head_count > NUMKONG_U32_MAX ||
        depth > NUMKONG_U32_MAX || !nk_size_mul_checked_(query_token_count, head_count, &rows))
        return nk_unexpected_dimensions_k;
    if (tasks_begin >= tasks_end || !depth) return nk_success_k;
    nk_size_t query_row_bytes, output_row_bytes, gradient_row_bytes, query_bytes, output_bytes, query_gradient_bytes,
        log_sum_exp_bytes;
    if (!nk_size_mul_checked_(head_count * depth, element_bytes, &query_row_bytes) ||
        !nk_size_mul_checked_(head_count * depth, sizeof(nk_f32_t), &output_row_bytes) ||
        !nk_size_mul_checked_(key_value_head_count * depth, sizeof(nk_f32_t), &gradient_row_bytes) ||
        !nk_size_span_checked_(query_token_count, query_stride, query_row_bytes, &query_bytes) ||
        !nk_size_span_checked_(query_token_count, output_stride, output_row_bytes, &output_bytes) ||
        !nk_size_span_checked_(query_token_count, query_gradient_stride, output_row_bytes, &query_gradient_bytes) ||
        !nk_size_mul_checked_(rows, sizeof(nk_f32_t), &log_sum_exp_bytes))
        return nk_unexpected_dimensions_k;

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t offsets[10] = {0};
    void *buffers[10];
    buffers[0] = nk_resolve_metal_(&call, queries, query_bytes, offsets);
    buffers[1] = nk_resolve_metal_(&call, packed, sizeof(nk_attention_packed_header_t), offsets + 1);
    buffers[2] = nk_resolve_metal_(&call, output, output_bytes, offsets + 2);
    buffers[3] = nk_resolve_metal_(&call, query_offsets, 2 * sizeof(nk_u32_t), offsets + 3);
    buffers[5] = nk_resolve_metal_(&call, log_sum_exp, log_sum_exp_bytes, offsets + 5);
    buffers[6] = nk_resolve_metal_(&call, output_gradient, output_bytes, offsets + 6);
    buffers[7] = nk_resolve_metal_(&call, query_gradient, query_gradient_bytes, offsets + 7);
    buffers[8] = nk_resolve_metal_(&call, key_gradient, gradient_row_bytes, offsets + 8);
    buffers[9] = nk_resolve_metal_(&call, value_gradient, gradient_row_bytes, offsets + 9);
    for (nk_size_t index = 0; index != 10; ++index)
        if (index != 4 && !buffers[index]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    nk_size_t const packed_bytes = nk_count_metal_(buffers[1], "length") - offsets[1];
    nk_size_t const offsets_bytes = nk_count_metal_(buffers[3], "length") - offsets[3];
    nk_size_t const key_gradient_bytes = nk_count_metal_(buffers[8], "length") - offsets[8];
    nk_size_t const value_gradient_bytes = nk_count_metal_(buffers[9], "length") - offsets[9];
    nk_size_t tasks;
    if (!nk_size_mul_checked_(offsets_bytes / sizeof(nk_u32_t) - 1, key_value_head_count, &tasks))
        tasks = NUMKONG_SIZE_MAX;
    if (tasks_end > tasks) tasks_end = tasks;
    if (tasks_begin >= tasks_end) return nk_abort_metal_(&call, nk_success_k);

    nk_attention_backward_arguments_metal_t arguments;
    nk_attention_arguments_metal_t *const attention = &arguments.attention;
    attention->heads = head_count, attention->kv_heads = key_value_head_count, attention->depth = depth;
    attention->query_tokens = query_token_count, attention->query_stride = query_stride;
    attention->output_stride = output_stride, attention->keys_before = keys_before, attention->keys_after = keys_after;
    attention->tasks_begin = tasks_begin, attention->tasks_end = tasks_end;
    attention->packed_bytes = packed_bytes, attention->offsets_bytes = offsets_bytes;
    attention->log_sum_exp_bytes = log_sum_exp_bytes, attention->capability = capability;
    attention->scale2 = scale * NUMKONG_F32_LOG2E_;
    attention->query_tensor_present = 0, attention->query_scales_stride = 0;
    arguments.query_gradient_stride = query_gradient_stride;
    arguments.key_value_gradient_stride = key_value_gradient_stride;
    arguments.key_value_gradient_bytes = nk_min_of_two(key_gradient_bytes, value_gradient_bytes);
    arguments.scale = scale;
    for (nk_size_t index = 0; index != 10; ++index)
        if (index != 4) nk_bind_metal_(call.encoder, buffers[index], offsets[index], index);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 4);

    void *const keys_pipeline = nk_pipeline_metal_(call.context, source, keys_kernel, language_version);
    void *const queries_pipeline = nk_pipeline_metal_(call.context, source, queries_kernel, language_version);
    if (!keys_pipeline || !queries_pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    // Threadgroups walk the window's tasks along x and tiles of keys or folded query rows along y
    nk_size_t const key_value_bytes = gradient_row_bytes / sizeof(nk_f32_t) * 2 * element_bytes;
    nk_size_t const key_blocks = nk_size_divide_round_up_(packed_bytes / key_value_bytes, tile);
    nk_size_t const query_blocks = nk_size_divide_round_up_(rows / key_value_head_count, tile);
    nk_size_t const key_groups = nk_max_of_two(1, nk_min_of_two(64, key_blocks));
    nk_size_t const query_groups = nk_max_of_two(1, nk_min_of_two(64, query_blocks));
    nk_size_t const window = tasks_end - tasks_begin;
    nk_metal_size_t const group_threads = {threads, 1, 1};
    nk_metal_size_t const keys_grid = {nk_min_of_two(window, 65535 / key_groups), key_groups, 1};
    nk_metal_size_t const queries_grid = {nk_min_of_two(window, 65535 / query_groups), query_groups, 1};
    nk_encoder_metal_(&call, keys_pipeline);
    nk_attention_encode_metal_(&call, keys_grid, group_threads);
    nk_encoder_metal_(&call, queries_pipeline);
    return nk_dispatch_metal_(&call, queries_grid, group_threads);
}

#pragma endregion Launchers

#if NUMKONG_TARGET_METAL

NUMKONG_API nk_status_t nk_attention_rope_f32_metal(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    return nk_attention_rope_launch_metal_("nk_attention_rope_f32_metal_kernel_", sizeof(nk_f32_t), x, cos, sin, y,
                                           rows, head_count, depth, x_stride, y_stride, stream);
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_metal(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                     nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                     nk_stream_t stream) {
    return nk_attention_rope_launch_metal_("nk_attention_rope_bf16_metal_kernel_", sizeof(nk_bf16_t), x, cos, sin, y,
                                           rows, head_count, depth, x_stride, y_stride, stream);
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_metal(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                     nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                     nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                     nk_stream_t stream) {
    return nk_attention_rope_launch_metal_("nk_attention_rope_e4m3_metal_kernel_", sizeof(nk_e4m3_t), x, cos, sin, y,
                                           rows, head_count, depth, x_stride, y_stride, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_metal(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_bf16_t const *keys,
                                                     nk_size_t key_stride, nk_bf16_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(
        nk_cap_metal_k, "nk_attention_directory_metal_kernel_", "nk_attention_pack_metal_kernel_", sizeof(nk_bf16_t),
        depth * sizeof(nk_bf16_t), 0, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
        nk_cross_operand_serial_(nk_bf16_k, keys, key_stride), key_stride,
        nk_cross_operand_serial_(nk_bf16_k, values, value_stride), value_stride, key_value_packed, tasks_begin,
        tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_bf16_metal_kernel_", "nk_attention_split_bf16_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_bf16_metal_kernel_", sizeof(nk_bf16_t), depth * sizeof(nk_bf16_t), 0, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_bf16_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_gradients_bf16_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_backward_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_backward_keys_bf16_metal_kernel_", "nk_attention_backward_queries_bf16_metal_kernel_", 8, 256,
        sizeof(nk_bf16_t), head_count, key_value_head_count, depth, query_offsets, query_token_count, scale,
        keys_before, keys_after, queries, query_stride, key_value_packed, output, output_gradient, output_stride,
        log_sum_exp, query_gradient, query_gradient_stride, key_gradient, value_gradient, key_value_gradient_stride,
        tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_f16_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_metal(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_f16_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                    nk_f16_t const *values, nk_size_t value_stride,
                                                    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(
        nk_cap_metal_k, "nk_attention_directory_metal_kernel_", "nk_attention_pack_metal_kernel_", sizeof(nk_f16_t),
        depth * sizeof(nk_f16_t), 0, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
        nk_cross_operand_serial_(nk_f16_k, keys, key_stride), key_stride,
        nk_cross_operand_serial_(nk_f16_k, values, value_stride), value_stride, key_value_packed, tasks_begin,
        tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_f16_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_f16_metal_kernel_", "nk_attention_split_f16_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_f16_metal_kernel_", sizeof(nk_f16_t), depth * sizeof(nk_f16_t), 0, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_f16_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_e4m3_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_metal(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_e4m3_t const *keys,
                                                     nk_size_t key_stride, nk_e4m3_t const *values,
                                                     nk_size_t value_stride, void *key_value_packed,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(
        nk_cap_metal_k, "nk_attention_directory_metal_kernel_", "nk_attention_pack_metal_kernel_", sizeof(nk_e4m3_t),
        depth * sizeof(nk_e4m3_t), 0, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
        nk_cross_operand_serial_(nk_e4m3_k, keys, key_stride), key_stride,
        nk_cross_operand_serial_(nk_e4m3_k, values, value_stride), value_stride, key_value_packed, tasks_begin,
        tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_e4m3_metal_kernel_", "nk_attention_split_e4m3_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_e4m3_metal_kernel_", sizeof(nk_e4m3_t), depth * sizeof(nk_e4m3_t), 0, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_e4m3_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_i8_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_metal(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_i8_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                   nk_i8_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_metal_kernel_",
                                           "nk_attention_pack_metal_kernel_", sizeof(nk_i8_t), depth * sizeof(nk_i8_t),
                                           0, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                           nk_cross_operand_serial_(nk_i8_k, keys, key_stride), key_stride,
                                           nk_cross_operand_serial_(nk_i8_k, values, value_stride), value_stride,
                                           key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_i8_metal(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_i8_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k, "nk_attention_packed_i8_metal_kernel_",
        "nk_attention_split_i8_metal_kernel_", "nk_attention_weights_i8_metal_kernel_",
        "nk_attention_merge_i8_metal_kernel_", sizeof(nk_i8_t), depth * sizeof(nk_i8_t), 0, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_i8_k, queries, query_stride), query_stride, key_value_packed, output, output_stride,
        log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_f16_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_metal(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_nvfp4_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                      nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 16 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_nvfp4_metal_kernel_",
                                           "nk_attention_pack_nvfp4_metal_kernel_", sizeof(nk_f16_t), depth / 2,
                                           depth / 16, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_nvfp4_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_nvfp4_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_nvfp4_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_nvfp4_metal_kernel_", "nk_attention_split_nvfp4_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_nvfp4_metal_kernel_", sizeof(nk_f16_t), depth / 2, depth / 16, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_nvfp4_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t), bytes);
    *bytes += nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_metal(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp4_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                      nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_mxfp4_metal_kernel_",
                                           "nk_attention_pack_mxfp4_metal_kernel_", sizeof(nk_bf16_t), depth / 2,
                                           depth / 32, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_mxfp4_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_mxfp4_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp4_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_mxfp4_metal_kernel_", "nk_attention_split_mxfp4_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_mxfp4_metal_kernel_", sizeof(nk_bf16_t), depth / 2, depth / 32, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_mxfp4_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes) {
    nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t), bytes);
    *bytes += nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_metal(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_metal(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_mxfp6e2m3_metal_kernel_",
                                           "nk_attention_pack_mxfp6e2m3_metal_kernel_", sizeof(nk_bf16_t), depth,
                                           depth / 32, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_mxfp6e2m3_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_mxfp6e2m3_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_mxfp6e2m3_metal_kernel_", "nk_attention_split_mxfp6e2m3_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_mxfp6e2m3_metal_kernel_", sizeof(nk_bf16_t), depth, depth / 32, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_mxfp6e2m3_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes) {
    nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t), bytes);
    *bytes += nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_metal(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_metal(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_mxfp6e3m2_metal_kernel_",
                                           "nk_attention_pack_mxfp6e3m2_metal_kernel_", sizeof(nk_bf16_t), depth,
                                           depth / 32, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_mxfp6e3m2_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_mxfp6e3m2_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_mxfp6e3m2_metal_kernel_", "nk_attention_split_mxfp6e3m2_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_mxfp6e3m2_metal_kernel_", sizeof(nk_bf16_t), depth, depth / 32, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_mxfp6e3m2_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes) {
    nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t), bytes);
    *bytes += nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_metal(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_metal(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_mxfp8e4m3_metal_kernel_",
                                           "nk_attention_pack_mxfp8e4m3_metal_kernel_", sizeof(nk_bf16_t), depth,
                                           depth / 32, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_mxfp8e4m3_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_mxfp8e4m3_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_mxfp8e4m3_metal_kernel_", "nk_attention_split_mxfp8e4m3_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_mxfp8e4m3_metal_kernel_", sizeof(nk_bf16_t), depth, depth / 32, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_mxfp8e4m3_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_metal(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_size_t token_count, nk_size_t segment_count,
                                                               nk_size_t *bytes) {
    nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t), bytes);
    *bytes += nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_metal(void const *key_value_packed,
                                                                  nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                  nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_metal_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_metal(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    return nk_attention_pack_launch_metal_(nk_cap_metal_k, "nk_attention_directory_mxfp8e5m2_metal_kernel_",
                                           "nk_attention_pack_mxfp8e5m2_metal_kernel_", sizeof(nk_bf16_t), depth,
                                           depth / 32, key_value_head_count, depth, key_offsets, key_lengths,
                                           segment_count, nk_cross_operand_serial_(nk_mxfp8e5m2_k, keys, key_stride),
                                           key_stride, nk_cross_operand_serial_(nk_mxfp8e5m2_k, values, value_stride),
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_metal(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_metal_, NUMKONG_METAL_LANGUAGE_3_1_, nk_cap_metal_k,
        "nk_attention_packed_mxfp8e5m2_metal_kernel_", "nk_attention_split_mxfp8e5m2_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_mxfp8e5m2_metal_kernel_", sizeof(nk_bf16_t), depth, depth / 32, head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after,
        nk_cross_operand_serial_(nk_mxfp8e5m2_k, queries, query_stride), query_stride, key_value_packed, output,
        output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_ATTENTION_METAL_H
