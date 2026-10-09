/**
 *  @file include/numkong/attention/apple10.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Ragged attention on Apple family 10 GPUs, launched from C.
 *
 *  @sa include/numkong/attention.h
 *  @sa include/numkong/attention/metal.h
 *  @sa include/numkong/attention/apple10.metal, the kernels this embeds
 *
 *  The @c metal pack and launch contract over @c matmul2d tiles on the Neural Accelerators, I8
 *  among them, as integer products sum exactly there. The kernels travel as their own `.metal`
 *  source, embedded after `metal.metal`, whose online softmax and decode path they share, and
 *  compile on the device at first use under Metal 4.0, the first with tensor operations.
 */
#ifndef NUMKONG_ATTENTION_APPLE10_H
#define NUMKONG_ATTENTION_APPLE10_H

#if NUMKONG_ARCH_METAL_
#if NUMKONG_TARGET_APPLE10
#include "numkong/attention/metal.h" // `nk_attention_launch_metal_`, `nk_attention_pack_launch_metal_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_attention_source_apple10_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/attention/metal.metal" suffix(, )
#embed "numkong/attention/apple10.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_apple10(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_apple10_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_bf16_t const *keys,
                                                       nk_size_t key_stride, nk_bf16_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(nk_cap_apple10_k, sizeof(nk_bf16_t), key_value_head_count, depth,
                                           key_offsets, key_lengths, segment_count, keys, key_stride, values,
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_bf16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_, nk_cap_apple10_k,
        "nk_attention_packed_bf16_apple10_kernel_", "nk_attention_split_bf16_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_bf16_metal_kernel_", sizeof(nk_bf16_t), head_count, key_value_head_count, depth,
        query_offsets, query_token_count, scale, keys_before, keys_after, queries, query_stride, key_value_packed,
        output, output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_f16_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_apple10(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_apple10_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_f16_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                      nk_size_t segment_count, nk_f16_t const *keys,
                                                      nk_size_t key_stride, nk_f16_t const *values,
                                                      nk_size_t value_stride, void *key_value_packed,
                                                      nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(nk_cap_apple10_k, sizeof(nk_f16_t), key_value_head_count, depth, key_offsets,
                                           key_lengths, segment_count, keys, key_stride, values, value_stride,
                                           key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_f16_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_f16_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(nk_attention_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_, nk_cap_apple10_k,
                                      "nk_attention_packed_f16_apple10_kernel_", "nk_attention_split_f16_metal_kernel_",
                                      NUMKONG_NULL, "nk_attention_merge_f16_metal_kernel_", sizeof(nk_f16_t),
                                      head_count, key_value_head_count, depth, query_offsets, query_token_count, scale,
                                      keys_before, keys_after, queries, query_stride, key_value_packed, output,
                                      output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_e4m3_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_apple10(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_apple10_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                       nk_size_t segment_count, nk_e4m3_t const *keys,
                                                       nk_size_t key_stride, nk_e4m3_t const *values,
                                                       nk_size_t value_stride, void *key_value_packed,
                                                       nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(nk_cap_apple10_k, sizeof(nk_e4m3_t), key_value_head_count, depth,
                                           key_offsets, key_lengths, segment_count, keys, key_stride, values,
                                           value_stride, key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_e4m3_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_, nk_cap_apple10_k,
        "nk_attention_packed_e4m3_apple10_kernel_", "nk_attention_split_e4m3_metal_kernel_", NUMKONG_NULL,
        "nk_attention_merge_e4m3_metal_kernel_", sizeof(nk_e4m3_t), head_count, key_value_head_count, depth,
        query_offsets, query_token_count, scale, keys_before, keys_after, queries, query_stride, key_value_packed,
        output, output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    return nk_attention_pack_size_metal_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_i8_t),
                                         bytes);
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_apple10(void const *key_value_packed,
                                                             nk_size_t *key_value_head_count, nk_size_t *depth,
                                                             nk_size_t *segments, nk_stream_t stream) {
    return nk_attention_packed_shape_metal_(key_value_packed, key_value_head_count, depth, segments, nk_cap_apple10_k,
                                            stream);
}

NUMKONG_API nk_status_t nk_attention_pack_i8_apple10(nk_size_t key_value_head_count, nk_size_t depth,
                                                     nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                     nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                     nk_i8_t const *values, nk_size_t value_stride,
                                                     void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                     nk_stream_t stream) {
    return nk_attention_pack_launch_metal_(nk_cap_apple10_k, sizeof(nk_i8_t), key_value_head_count, depth, key_offsets,
                                           key_lengths, segment_count, keys, key_stride, values, value_stride,
                                           key_value_packed, tasks_begin, tasks_end, stream);
}

NUMKONG_API nk_status_t nk_attention_packed_i8_apple10(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, nk_i8_t const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output, nk_size_t output_stride,
    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    return nk_attention_launch_metal_(
        nk_attention_source_apple10_, NUMKONG_METAL_LANGUAGE_4_0_, nk_cap_apple10_k,
        "nk_attention_packed_i8_apple10_kernel_", "nk_attention_split_i8_metal_kernel_",
        "nk_attention_weights_i8_metal_kernel_", "nk_attention_merge_i8_metal_kernel_", sizeof(nk_i8_t), head_count,
        key_value_head_count, depth, query_offsets, query_token_count, scale, keys_before, keys_after, queries,
        query_stride, key_value_packed, output, output_stride, log_sum_exp, tasks_begin, tasks_end, stream);
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_TARGET_APPLE10
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_ATTENTION_APPLE10_H
