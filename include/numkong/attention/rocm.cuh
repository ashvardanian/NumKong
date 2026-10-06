/**
 *  @file include/numkong/attention/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of packed attention: the pack and attention launches, the generators every
 *      ROCm capability instantiates its exports with, and the baseline @c rocm exports, over the
 *      kernels of `attention/simt.cuh`.
 *
 *  @sa include/numkong/attention/simt.cuh
 *  @sa include/numkong/attention/cuda.cuh
 */
#ifndef NUMKONG_ATTENTION_ROCM_CUH
#define NUMKONG_ATTENTION_ROCM_CUH

#include "numkong/rocm.cuh"
#include "numkong/attention/simt.cuh"

#if NUMKONG_ARCH_ROCM_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Launchers

/** Launches the directory writer, recording the packing @p capability, for a window starting at
 *  task 0, then @p payload_kernel over the window. */
NUMKONG_INLINE nk_status_t nk_attention_pack_launch_rocm_(void const *payload_kernel, nk_capability_t capability,
                                                          nk_size_t element_bytes, void const *keys, void const *values,
                                                          nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_offsets,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                          nk_size_t key_stride, nk_size_t value_stride, void *packed,
                                                          nk_size_t task_begin, nk_size_t task_end, void *stream) {
    if ((nk_size_t)packed & 15) return nk_misaligned_k;
    nk_size_t const row_bytes = nk_size_round_up_to_multiple_(depth * element_bytes, nk_attention_step_bytes_k);
    nk_size_t const directory_bytes = nk_attention_pack_directory_size_(segment_count);
    if (task_begin == 0) {
        void *directory_arguments[8] = {&packed,
                                        &key_value_head_count,
                                        &depth,
                                        &segment_lengths,
                                        &segment_count,
                                        (void *)&row_bytes,
                                        (void *)&directory_bytes,
                                        (void *)&capability};
        nk_status_t const status = nk_launch_rocm_((void const *)nk_attention_pack_directory_simt_kernel_, 1,
                                                   nk_attention_threads_k, directory_arguments, 0, stream);
        if (status != nk_success_k) return status;
    }
    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const end = task_end < total_tasks ? task_end : total_tasks;
    if (task_begin >= end) return nk_success_k;
    void *payload_arguments[12] = {(void *)&keys,    (void *)&values,  &key_value_head_count, &depth,
                                   &segment_offsets, &segment_lengths, &segment_count,        &key_stride,
                                   &value_stride,    &packed,          &task_begin,           (void *)&end};
    return nk_launch_rocm_(payload_kernel, end - task_begin < 65535 ? end - task_begin : 65535,
                           nk_attention_pack_threads_k, payload_arguments, 0, stream);
}

/** Validates the contract and launches @p kernel with as many blocks as stay resident. */
NUMKONG_INLINE nk_status_t nk_attention_launch_rocm_(
    void const *kernel, nk_capability_t capability, void const *queries, void const *packed, nk_f32_t *output,
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_attention_mask_t mask, nk_i64_t diagonal_offset,
    nk_size_t window, nk_size_t task_start, nk_size_t task_count, void *stream) {
    if (((nk_size_t)packed & 15) || (((nk_size_t)output | output_stride) & 3)) return nk_misaligned_k;
    if (key_value_head_count == 0 || head_count % key_value_head_count != 0) return nk_unexpected_dimensions_k;
    if (task_count == 0 || depth == 0) return nk_success_k;
    nk_attention_arguments_t arguments = nk_attention_arguments_init_(
        queries, packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride, output_stride,
        scale, 1, 1, mask, diagonal_offset, window, task_start, task_count);
    return nk_launch_resident_rocm_(kernel, nk_attention_threads_k, 0, 0, NUMKONG_SIZE_MAX, &arguments, stream);
}

#pragma endregion Launchers

#pragma region Attention Macros

/** Generates a shape accessor that copies a device pack's header back and checks its capability. */
#define nk_define_attention_packed_shape_rocm_(input_type_name, isa_suffix)                                    \
    NUMKONG_API nk_status_t nk_attention_packed_shape_##input_type_name##_##isa_suffix(                        \
        void const *key_value_packed, nk_size_t *heads, nk_size_t *depth, nk_size_t *segments, void *stream) { \
        nk_attention_packed_header_t header;                                                                   \
        nk_status_t const status = nk_read_rocm_(&header, key_value_packed, sizeof(header), stream);           \
        if (status != nk_success_k) return status;                                                             \
        if (header.capability != nk_cap_##isa_suffix##_k) return nk_pack_mismatch_k;                           \
        *heads = header.heads, *depth = header.depth, *segments = header.segments;                             \
        return nk_success_k;                                                                                   \
    }

/** Generates a device pack: the directory, recording the packing @p isa_suffix, when the window
 *  starts at task 0, then the kernel of @c nk_define_attention_pack_kernel_simt_. */
#define nk_define_attention_pack_rocm_(input_type_name, isa_suffix, input_value_type)                                  \
    nk_define_attention_pack_kernel_simt_(input_type_name, isa_suffix, input_value_type) NUMKONG_API nk_status_t       \
    nk_attention_pack_##input_type_name##_##isa_suffix(                                                                \
        nk_##input_value_type##_t const *keys, nk_##input_value_type##_t const *values,                                \
        nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *segment_offsets,                              \
        nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,        \
        void *key_value_packed, nk_size_t task_begin, nk_size_t task_end, void *stream) {                              \
        return nk_attention_pack_launch_rocm_(                                                                         \
            (void const *)nk_attention_pack_##input_type_name##_##isa_suffix##_kernel_, nk_cap_##isa_suffix##_k,       \
            sizeof(nk_##input_value_type##_t), keys, values, key_value_head_count, depth, segment_offsets,             \
            segment_lengths, segment_count, key_stride, value_stride, key_value_packed, task_begin, task_end, stream); \
    }

/**
 *  @brief Generates both public attention entry points over the fallback kernel of
 *      @c nk_define_attention_fallback_kernel_simt_, each passing its @c nk_attention_mask_t.
 *
 *  The pack must come from the same capability's pack kernel: the host can't read its device header
 *  without waiting on the stream, so the entry points trust it.
 */
#define nk_define_attention_baseline_packed_rocm_(input_type_name, isa_suffix, input_value_type)                       \
    nk_define_attention_fallback_kernel_simt_(input_type_name, isa_suffix, input_value_type) NUMKONG_API nk_status_t   \
    nk_attention_bidirectional_packed_##input_type_name##_##isa_suffix(                                                \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,          \
        nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,   \
        void *stream) {                                                                                                \
        return nk_attention_launch_rocm_(                                                                              \
            (void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_, nk_cap_##isa_suffix##_k,     \
            queries, key_value_packed, output, head_count, key_value_head_count, depth, query_offsets, query_stride,   \
            output_stride, scale, nk_attention_mask_bidirectional_k, 0, 0, task_start, task_count, stream);            \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_causal_packed_##input_type_name##_##isa_suffix(                               \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,          \
        nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window,   \
        nk_size_t task_start, nk_size_t task_count, void *stream) {                                                    \
        return nk_attention_launch_rocm_((void const *)nk_attention_packed_##input_type_name##_##isa_suffix##_kernel_, \
                                         nk_cap_##isa_suffix##_k, queries, key_value_packed, output, head_count,       \
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride,      \
                                         scale, nk_attention_mask_causal_k, diagonal_offset, window, task_start,       \
                                         task_count, stream);                                                          \
    }

/** Every attention entry of one dtype on the ROCm baseline @p isa_suffix, with its own pack. */
#define nk_define_attention_baseline_rocm_(input_type_name, isa_suffix, input_value_type, element_bytes) \
    nk_define_attention_pack_size_simt_(input_type_name, isa_suffix, element_bytes)                      \
    nk_define_attention_packed_shape_rocm_(input_type_name, isa_suffix)                                  \
    nk_define_attention_pack_rocm_(input_type_name, isa_suffix, input_value_type)                        \
    nk_define_attention_baseline_packed_rocm_(input_type_name, isa_suffix, input_value_type)

/**
 *  @brief Generates both public attention entry points over the narrow, wide and fallback kernels of
 *      @c nk_define_attention_packed_kernels_simt_, each passing its @c nk_attention_mask_t to the
 *      capability's launch.
 *
 *  The pack must come from the same capability's pack kernel, which the entry points trust, as
 *  @c nk_define_attention_baseline_packed_rocm_ explains.
 *
 *  @param[in] launch_fn The launch, taking the narrow, wide and fallback kernels in that order.
 *  @param[in] score_scale Undoes the power of two that converting Q and K puts on scores, or 1.
 *  @param[in] output_scale Undoes the power of two that converting V puts on the output, or 1.
 */
#define nk_define_attention_packed_rocm_(input_type_name, isa_suffix, tile, launch_fn, input_value_type, epilogue,     \
                                         scores_fn, values_mma_fn, weights_fn, score_scale, output_scale)              \
    nk_define_attention_packed_kernels_simt_(input_type_name, isa_suffix, tile, input_value_type, epilogue, scores_fn, \
                                             values_mma_fn, weights_fn) NUMKONG_API nk_status_t                        \
    nk_attention_bidirectional_packed_##input_type_name##_##isa_suffix(                                                \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,          \
        nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_size_t task_start, nk_size_t task_count,   \
        void *stream) {                                                                                                \
        return launch_fn((void const *)nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_,          \
                         (void const *)nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_,            \
                         (void const *)nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_,        \
                         nk_##input_value_type##_k, queries, key_value_packed, output, head_count,                     \
                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale, score_scale,  \
                         output_scale, nk_attention_mask_bidirectional_k, 0, 0, task_start, task_count, stream);       \
    }                                                                                                                  \
    NUMKONG_API nk_status_t nk_attention_causal_packed_##input_type_name##_##isa_suffix(                               \
        nk_##input_value_type##_t const *queries, void const *key_value_packed, nk_f32_t *output,                      \
        nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,          \
        nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, nk_i64_t diagonal_offset, nk_size_t window,   \
        nk_size_t task_start, nk_size_t task_count, void *stream) {                                                    \
        return launch_fn((void const *)nk_attention_packed_##input_type_name##_narrow_##isa_suffix##_kernel_,          \
                         (void const *)nk_attention_packed_##input_type_name##_wide_##isa_suffix##_kernel_,            \
                         (void const *)nk_attention_packed_##input_type_name##_fallback_##isa_suffix##_kernel_,        \
                         nk_##input_value_type##_k, queries, key_value_packed, output, head_count,                     \
                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale, score_scale,  \
                         output_scale, nk_attention_mask_causal_k, diagonal_offset, window, task_start, task_count,    \
                         stream);                                                                                      \
    }

#pragma endregion Attention Macros

#pragma region Instantiations

#if NUMKONG_TARGET_ROCM
nk_define_attention_baseline_rocm_(bf16, rocm, bf16, 2)
nk_define_attention_baseline_rocm_(e4m3, rocm, e4m3, 1)
nk_define_attention_baseline_rocm_(i8, rocm, i8, 1)
#endif // NUMKONG_TARGET_ROCM

#pragma endregion Instantiations

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_ATTENTION_ROCM_CUH
