/**
 *  @file include/numkong/cast/metal.h
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief Bulk and block-scaled casts on the SIMT cores of every Apple GPU, launched from C.
 *
 *  @sa include/numkong/cast.h
 *  @sa include/numkong/cast/metal.metal, the kernels this embeds
 *  @sa include/numkong/cast/cuda.cuh, the CUDA sibling
 *
 *  The signature matches the @c cuda capability's, with an @c id<MTLCommandQueue> as the stream.
 *  The launcher plans the hub of a bulk cast the way @c nk_cast_elementwise_serial_ picks it, and
 *  a block-scaled cast deriving its tensor scale commits an abs-max pass before the chunks.
 */
#ifndef NUMKONG_CAST_METAL_H
#define NUMKONG_CAST_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h"       // `nk_metal_call_t`
#include "numkong/cast/serial.h" // `nk_cast_plan_serial_`, `nk_cast_operand_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_cast_source_metal_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/cast/metal.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Threads of every cast threadgroup, and the most threadgroups one launch loops over. */
enum { nk_cast_threads_metal_k = 256, nk_cast_groups_metal_k = 65535 };

/** The launch record of a block-scaled cast, laid out as the kernels' record of the same name. A
 *  plain side has an unknown scale type and blocks of one value, and a tensor scale not applied an
 *  unknown tensor scale type. */
typedef struct {
    nk_u64_t count, chunks;
    nk_u32_t from_dtype, from_scale_dtype, from_tensor_scale_dtype, from_block;
    nk_u32_t to_dtype, to_scale_dtype, to_tensor_scale_dtype, to_block;
} nk_cast_block_scaled_arguments_metal_t;

nk_static_assert_(sizeof(nk_cast_plan_t) == 32, nk_cast_plan_must_be_32_bytes);
nk_static_assert_(sizeof(nk_cast_block_scaled_arguments_metal_t) == 48,
                  nk_cast_block_scaled_arguments_metal_must_be_48_bytes);

/** The bytes that @p count values of @p dtype span, rounded up to whole bytes, or zero when the
 *  product overflows. */
NUMKONG_CONSTEXPR int nk_dtype_bytes_metal_(nk_dtype_t dtype, nk_size_t count, nk_size_t *bytes) {
    nk_size_t bits = 0;
    if (!nk_size_mul_checked_(count, nk_dtype_bits(dtype), &bits)) return 0;
    *bytes = nk_size_divide_round_up_(bits, NUMKONG_BITS_PER_BYTE);
    return 1;
}

/**
 *  @brief Encodes @p kernel over @p threads threads, at most as many threadgroups as a launch
 *      spreads its loop over, then commits it.
 *  @param[in] pointers Host addresses of the buffers bound in slots from zero, null for one the
 *      kernel leaves unread, which binds the first buffer that is not in its place.
 *  @param[in] bytes The bytes each buffer spans.
 */
NUMKONG_INLINE nk_status_t nk_cast_call_metal_(char const *kernel, void const *const *pointers, nk_size_t const *bytes,
                                               nk_size_t buffers_count, void const *arguments,
                                               nk_size_t arguments_bytes, nk_size_t threads, nk_stream_t stream) {
    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    void *buffers[6], *stand_in = NUMKONG_NULL;
    nk_size_t offsets[6], stand_in_offset = 0;
    for (nk_size_t index = 0; index != buffers_count; ++index) {
        if (!pointers[index]) continue;
        buffers[index] = nk_resolve_metal_(&call, pointers[index], bytes[index], offsets + index);
        if (!buffers[index]) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
        if (!stand_in) stand_in = buffers[index], stand_in_offset = offsets[index];
    }
    for (nk_size_t index = 0; index != buffers_count; ++index)
        if (!pointers[index]) buffers[index] = stand_in, offsets[index] = stand_in_offset;
    if (!stand_in) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, nk_cast_source_metal_, kernel, NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);
    for (nk_size_t index = 0; index != buffers_count; ++index)
        nk_bind_metal_(call.encoder, buffers[index], offsets[index], index);
    nk_bind_bytes_metal_(call.encoder, arguments, arguments_bytes, buffers_count);
    nk_size_t groups = nk_size_divide_round_up_(threads, nk_cast_threads_metal_k);
    if (groups > nk_cast_groups_metal_k) groups = nk_cast_groups_metal_k;
    nk_metal_size_t const grid = {groups, 1, 1}, group = {nk_cast_threads_metal_k, 1, 1};
    return nk_dispatch_metal_(&call, grid, group);
}

/** Validates a cast with a block-scaled side and commits its launches: a destination tensor scale
 *  adds the abs-max pass before the chunks, of which at least one thread runs, so an empty cast
 *  still writes the tensor scale it derives. */
NUMKONG_INLINE nk_status_t nk_cast_block_scaled_launch_metal_(nk_cast_operand_t const *from,
                                                              nk_block_scaled_format_t const *from_format,
                                                              nk_cast_operand_t const *to,
                                                              nk_block_scaled_format_t const *to_format,
                                                              nk_size_t count, nk_stream_t stream) {
    int const from_plain = from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0;
    int const to_plain = to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0;
    if ((!from_plain && from_format->block_size > 32) || (!to_plain && to_format->block_size > 32))
        return nk_unexpected_dimensions_k;
    if (((nk_size_t)from->codes & (nk_dtype_alignment_serial_(from_format->element_dtype) - 1)) ||
        ((nk_size_t)to->codes & (nk_dtype_alignment_serial_(to_format->element_dtype) - 1)))
        return nk_misaligned_k;
    nk_cast_block_scaled_arguments_metal_t arguments;
    arguments.count = count;
    arguments.from_dtype = from_format->element_dtype, arguments.to_dtype = to_format->element_dtype;
    arguments.from_scale_dtype = from_plain ? nk_dtype_unknown_k : from_format->scale_dtype;
    arguments.to_scale_dtype = to_plain ? nk_dtype_unknown_k : to_format->scale_dtype;
    arguments.from_tensor_scale_dtype = from->tensor_scale ? nk_f32_k : nk_dtype_unknown_k;
    arguments.to_tensor_scale_dtype = to->tensor_scale ? nk_f32_k : nk_dtype_unknown_k;
    arguments.from_block = from_plain ? 1u : (nk_u32_t)from_format->block_size;
    arguments.to_block = to_plain ? 1u : (nk_u32_t)to_format->block_size;
    arguments.chunks = nk_size_divide_round_up_(
        count, arguments.from_block > arguments.to_block ? arguments.from_block : arguments.to_block);
    if (!count && !to->tensor_scale) return nk_success_k;

    nk_size_t bytes[6] = {0, 0, sizeof(nk_f32_t), 0, 0, sizeof(nk_f32_t)};
    void const *const pointers[6] = {from->codes, from->scales, from->tensor_scale,
                                     to->codes,   to->scales,   to->tensor_scale};
    if (!nk_dtype_bytes_metal_(from_format->element_dtype, count, bytes + 0) ||
        !nk_dtype_bytes_metal_(to_format->element_dtype, count, bytes + 3))
        return nk_unexpected_dimensions_k;
    bytes[1] = nk_size_divide_round_up_(count, arguments.from_block);
    bytes[4] = nk_size_divide_round_up_(count, arguments.to_block);
    nk_status_t status = nk_success_k;
    if (to->tensor_scale && count)
        status = nk_cast_call_metal_("nk_cast_block_scaled_amax_metal_kernel_", pointers, bytes, 6, &arguments,
                                     sizeof(arguments), count, stream);
    if (status != nk_success_k) return status;
    return nk_cast_call_metal_("nk_cast_block_scaled_chunks_metal_kernel_", pointers, bytes, 6, &arguments,
                               sizeof(arguments), arguments.chunks ? arguments.chunks : 1, stream);
}

#if NUMKONG_TARGET_METAL
NUMKONG_API nk_status_t nk_cast_metal(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                      nk_size_t count, nk_stream_t stream) {
    if (nk_dtype_is_block_scaled(from_dtype) || nk_dtype_is_block_scaled(to_dtype)) {
        nk_block_scaled_format_t const from_format = nk_block_scaled_format_of_dtype(from_dtype);
        nk_block_scaled_format_t const to_format = nk_block_scaled_format_of_dtype(to_dtype);
        nk_cast_operand_t const source = nk_cast_operand_serial_(from_dtype, from);
        nk_cast_operand_t const target = nk_cast_operand_serial_(to_dtype, to);
        return nk_cast_block_scaled_launch_metal_(&source, &from_format, &target, &to_format, count, stream);
    }
    if (((nk_size_t)from & (nk_dtype_alignment_serial_(from_dtype) - 1)) ||
        ((nk_size_t)to & (nk_dtype_alignment_serial_(to_dtype) - 1)))
        return nk_misaligned_k;
    nk_cast_plan_t plan;
    nk_size_t bytes[2];
    if (!nk_dtype_bytes_metal_(from_dtype, count, bytes + 0) || !nk_dtype_bytes_metal_(to_dtype, count, bytes + 1))
        return nk_unexpected_dimensions_k;
    if (!nk_cast_plan_serial_(from_dtype, to_dtype, count, (nk_size_t)from | (nk_size_t)to, &plan)) return nk_success_k;
    void const *const pointers[2] = {from, to};
    return nk_cast_call_metal_("nk_cast_metal_kernel_", pointers, bytes, 2, &plan, sizeof(plan), plan.units, stream);
}
#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_CAST_METAL_H
