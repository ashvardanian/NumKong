/**
 *  @file include/numkong/each/metal.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief SwiGLU, grouped RMSNorm and its downcasting form on the SIMT cores of every Apple GPU,
 *      launched from C.
 *
 *  @sa include/numkong/each.h
 *  @sa include/numkong/each/metal.metal, the kernels this embeds
 *  @sa include/numkong/each/cuda.cuh, the CUDA sibling
 *
 *  Every signature matches the @c cuda capability's, with an @c id<MTLCommandQueue> as the stream.
 *  Each launcher picks the kernel for a present or missing @c up or @c gamma.
 */
#ifndef NUMKONG_EACH_METAL_H
#define NUMKONG_EACH_METAL_H

#if NUMKONG_ARCH_METAL_
#include "numkong/metal.h" // `nk_metal_call_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The MSL source of every kernel below, compiled once per device. */
static char const nk_each_source_metal_[] = {
#embed "numkong/types.metal" suffix(, )
#embed "numkong/each/metal.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** Threads of a SwiGLU threadgroup, and the most an RMSNorm one takes, as in the kernels. */
enum { nk_each_threads_metal_k = 256 };

/** The launch record of SwiGLU, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t gate_stride, up_stride, y_stride;
    nk_u32_t columns;
    nk_f32_t gate_scale, output_scale;
} nk_each_swiglu_arguments_metal_t;

/** The launch record of RMSNorm, laid out as the kernels' record of the same name. */
typedef struct {
    nk_u64_t x_stride, y_stride;
    nk_u32_t columns;
    nk_f32_t epsilon;
} nk_each_rmsnorm_arguments_metal_t;

nk_static_assert_(sizeof(nk_each_swiglu_arguments_metal_t) == 40, nk_each_swiglu_arguments_metal_must_be_40_bytes);
nk_static_assert_(sizeof(nk_each_rmsnorm_arguments_metal_t) == 24, nk_each_rmsnorm_arguments_metal_must_be_24_bytes);

/** Validates the contract and encodes SwiGLU, one thread per 16-byte chunk of a row, through
 *  @p gated_kernel, or @p ungated_kernel when @p up is NULL. */
NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_metal_(char const *gated_kernel, char const *ungated_kernel,
                                                        nk_size_t value_bytes, void const *gate, void const *up,
                                                        void *y, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                                        nk_f32_t gate_scale, nk_f32_t output_scale,
                                                        nk_stream_t stream) {
    if ((((nk_size_t)gate) | gate_stride | ((nk_size_t)up) | (up ? up_stride : 0) | ((nk_size_t)y) | y_stride) &
        (value_bytes - 1))
        return nk_misaligned_k;
    if (rows == 0 || columns == 0) return nk_success_k;
    nk_size_t row_bytes, gate_bytes, up_bytes = 0, y_bytes;
    if (rows > 0xFFFFFFFFu || columns > 0xFFFFFFFFu || !nk_size_mul_checked_(columns, value_bytes, &row_bytes) ||
        !nk_size_span_checked_(rows, gate_stride, row_bytes, &gate_bytes) ||
        (up && !nk_size_span_checked_(rows, up_stride, row_bytes, &up_bytes)) ||
        !nk_size_span_checked_(rows, y_stride, row_bytes, &y_bytes))
        return nk_unexpected_dimensions_k;

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t gate_offset = 0, up_offset = 0, y_offset = 0;
    void *const gate_buffer = nk_resolve_metal_(&call, gate, gate_bytes, &gate_offset);
    void *const up_buffer = up ? nk_resolve_metal_(&call, up, up_bytes, &up_offset) : gate_buffer;
    void *const y_buffer = nk_resolve_metal_(&call, y, y_bytes, &y_offset);
    if (!gate_buffer || !up_buffer || !y_buffer) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, nk_each_source_metal_, up ? gated_kernel : ungated_kernel,
                                              NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    nk_each_swiglu_arguments_metal_t arguments;
    arguments.gate_stride = gate_stride, arguments.up_stride = up_stride, arguments.y_stride = y_stride;
    arguments.columns = (nk_u32_t)columns, arguments.gate_scale = gate_scale, arguments.output_scale = output_scale;
    nk_bind_metal_(call.encoder, gate_buffer, gate_offset, 0);
    nk_bind_metal_(call.encoder, up_buffer, up ? up_offset : gate_offset, 1);
    nk_bind_metal_(call.encoder, y_buffer, y_offset, 2);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 3);
    nk_size_t const chunks = nk_size_divide_round_up_(columns, 16 / value_bytes);
    nk_metal_size_t const grid = {nk_size_divide_round_up_(chunks, nk_each_threads_metal_k), rows, 1};
    nk_metal_size_t const group = {nk_each_threads_metal_k, 1, 1};
    return nk_dispatch_metal_(&call, grid, group);
}

/** Validates the contract and encodes RMSNorm of @p input_bytes values into @p output_bytes ones,
 *  one threadgroup per vector, sized to the 16-byte chunks of its output, through
 *  @p scaled_kernel, or @p unscaled_kernel when @p gamma is NULL. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_metal_(char const *scaled_kernel, char const *unscaled_kernel,
                                                         nk_size_t input_bytes, nk_size_t output_bytes, void const *x,
                                                         nk_f32_t const *gamma, void *y, nk_size_t rows,
                                                         nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                         nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    if ((((nk_size_t)x) | x_stride) & (input_bytes - 1) || (((nk_size_t)y) | y_stride) & (output_bytes - 1) ||
        ((nk_size_t)gamma & 3))
        return nk_misaligned_k;
    if (rows == 0 || groups == 0 || columns == 0) return nk_success_k;
    nk_size_t vector_values, x_row_bytes, y_row_bytes, x_bytes, y_bytes, gamma_bytes;
    if (rows > 0xFFFFFFFFu || groups > 0xFFFFFFFFu / nk_each_threads_metal_k || columns > 0xFFFFFFFFu ||
        !nk_size_mul_checked_(groups, columns, &vector_values) ||
        !nk_size_mul_checked_(vector_values, input_bytes, &x_row_bytes) ||
        !nk_size_mul_checked_(vector_values, output_bytes, &y_row_bytes) ||
        !nk_size_span_checked_(rows, x_stride, x_row_bytes, &x_bytes) ||
        !nk_size_span_checked_(rows, y_stride, y_row_bytes, &y_bytes) ||
        !nk_size_mul_checked_(columns, sizeof(nk_f32_t), &gamma_bytes))
        return nk_unexpected_dimensions_k;

    nk_metal_call_t call;
    nk_status_t const status = nk_enter_metal_(stream, &call);
    if (status != nk_success_k) return status;
    nk_size_t x_offset = 0, gamma_offset = 0, y_offset = 0;
    void *const x_buffer = nk_resolve_metal_(&call, x, x_bytes, &x_offset);
    void *const gamma_buffer = gamma ? nk_resolve_metal_(&call, gamma, gamma_bytes, &gamma_offset) : x_buffer;
    void *const y_buffer = nk_resolve_metal_(&call, y, y_bytes, &y_offset);
    if (!x_buffer || !gamma_buffer || !y_buffer) return nk_abort_metal_(&call, nk_device_memory_mismatch_k);
    void *const pipeline = nk_pipeline_metal_(call.context, nk_each_source_metal_,
                                              gamma ? scaled_kernel : unscaled_kernel, NUMKONG_METAL_LANGUAGE_3_1_);
    if (!pipeline) return nk_abort_metal_(&call, nk_device_code_mismatch_k);
    nk_encoder_metal_(&call, pipeline);

    nk_each_rmsnorm_arguments_metal_t arguments;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride;
    arguments.columns = (nk_u32_t)columns, arguments.epsilon = epsilon;
    nk_bind_metal_(call.encoder, x_buffer, x_offset, 0);
    nk_bind_metal_(call.encoder, gamma_buffer, gamma ? gamma_offset : x_offset, 1);
    nk_bind_metal_(call.encoder, y_buffer, y_offset, 2);
    nk_bind_bytes_metal_(call.encoder, &arguments, sizeof(arguments), 3);
    nk_size_t const chunks = nk_size_divide_round_up_(columns, 16 / output_bytes);
    nk_size_t const threads = chunks < nk_each_threads_metal_k ? nk_size_round_up_to_multiple_(chunks, 32)
                                                               : nk_each_threads_metal_k;
    nk_metal_size_t const grid = {groups, rows, 1}, group = {threads, 1, 1};
    return nk_dispatch_metal_(&call, grid, group);
}

#if NUMKONG_TARGET_METAL
NUMKONG_API nk_status_t nk_each_swiglu_f32_metal(nk_f32_t const *gate, nk_f32_t const *up, nk_f32_t *y, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                                 nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                                 nk_stream_t stream) {
    return nk_each_swiglu_launch_metal_("nk_each_swiglu_f32_gated_metal_kernel_",
                                        "nk_each_swiglu_f32_ungated_metal_kernel_", sizeof(nk_f32_t), gate, up, y, rows,
                                        columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream);
}

NUMKONG_API nk_status_t nk_each_swiglu_f16_metal(nk_f16_t const *gate, nk_f16_t const *up, nk_f16_t *y, nk_size_t rows,
                                                 nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                                 nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                                 nk_stream_t stream) {
    return nk_each_swiglu_launch_metal_("nk_each_swiglu_f16_gated_metal_kernel_",
                                        "nk_each_swiglu_f16_ungated_metal_kernel_", sizeof(nk_f16_t), gate, up, y, rows,
                                        columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream);
}

NUMKONG_API nk_status_t nk_each_swiglu_bf16_metal(nk_bf16_t const *gate, nk_bf16_t const *up, nk_bf16_t *y,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t gate_stride,
                                                  nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,
                                                  nk_f32_t output_scale, nk_stream_t stream) {
    return nk_each_swiglu_launch_metal_(
        "nk_each_swiglu_bf16_gated_metal_kernel_", "nk_each_swiglu_bf16_ungated_metal_kernel_", sizeof(nk_bf16_t), gate,
        up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream);
}

NUMKONG_API nk_status_t nk_each_swiglu_e4m3_metal(nk_e4m3_t const *gate, nk_e4m3_t const *up, nk_e4m3_t *y,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t gate_stride,
                                                  nk_size_t up_stride, nk_size_t y_stride, nk_f32_t gate_scale,
                                                  nk_f32_t output_scale, nk_stream_t stream) {
    return nk_each_swiglu_launch_metal_(
        "nk_each_swiglu_e4m3_gated_metal_kernel_", "nk_each_swiglu_e4m3_ungated_metal_kernel_", sizeof(nk_e4m3_t), gate,
        up, y, rows, columns, gate_stride, up_stride, y_stride, gate_scale, output_scale, stream);
}

NUMKONG_API nk_status_t nk_each_rmsnorm_f32_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_f32_t *y, nk_size_t rows,
                                                  nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                  nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmsnorm_f32_scaled_metal_kernel_", "nk_each_rmsnorm_f32_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_f32_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmsnorm_f16_metal(nk_f16_t const *x, nk_f32_t const *gamma, nk_f16_t *y, nk_size_t rows,
                                                  nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                  nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmsnorm_f16_scaled_metal_kernel_", "nk_each_rmsnorm_f16_unscaled_metal_kernel_", sizeof(nk_f16_t),
        sizeof(nk_f16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmsnorm_bf16_metal(nk_bf16_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmsnorm_bf16_scaled_metal_kernel_", "nk_each_rmsnorm_bf16_unscaled_metal_kernel_", sizeof(nk_bf16_t),
        sizeof(nk_bf16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmsnorm_e4m3_metal(nk_e4m3_t const *x, nk_f32_t const *gamma, nk_e4m3_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmsnorm_e4m3_scaled_metal_kernel_", "nk_each_rmsnorm_e4m3_unscaled_metal_kernel_", sizeof(nk_e4m3_t),
        sizeof(nk_e4m3_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_bf16_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_bf16_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_bf16_scaled_metal_kernel_", "nk_each_rmscast_bf16_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_bf16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_f16_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_f16_t *y, nk_size_t rows,
                                                  nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                  nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_f16_scaled_metal_kernel_", "nk_each_rmscast_f16_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_f16_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_e4m3_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_e4m3_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_e4m3_scaled_metal_kernel_", "nk_each_rmscast_e4m3_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_e4m3_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_e5m2_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_e5m2_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_e5m2_scaled_metal_kernel_", "nk_each_rmscast_e5m2_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_e5m2_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_e2m3_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_e2m3_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_e2m3_scaled_metal_kernel_", "nk_each_rmscast_e2m3_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_e2m3_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_e3m2_metal(nk_f32_t const *x, nk_f32_t const *gamma, nk_e3m2_t *y,
                                                   nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                   nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_(
        "nk_each_rmscast_e3m2_scaled_metal_kernel_", "nk_each_rmscast_e3m2_unscaled_metal_kernel_", sizeof(nk_f32_t),
        sizeof(nk_e3m2_t), x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_i8_metal(nk_i32_t const *x, nk_f32_t const *gamma, nk_i8_t *y, nk_size_t rows,
                                                 nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                 nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_("nk_each_rmscast_i8_scaled_metal_kernel_",
                                         "nk_each_rmscast_i8_unscaled_metal_kernel_", sizeof(nk_i32_t), sizeof(nk_i8_t),
                                         x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}

NUMKONG_API nk_status_t nk_each_rmscast_u8_metal(nk_u32_t const *x, nk_f32_t const *gamma, nk_u8_t *y, nk_size_t rows,
                                                 nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                 nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream) {
    return nk_each_rmsnorm_launch_metal_("nk_each_rmscast_u8_scaled_metal_kernel_",
                                         "nk_each_rmscast_u8_unscaled_metal_kernel_", sizeof(nk_u32_t), sizeof(nk_u8_t),
                                         x, gamma, y, rows, groups, columns, x_stride, y_stride, epsilon, stream);
}
#endif // NUMKONG_TARGET_METAL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_ARCH_METAL_
#endif // NUMKONG_EACH_METAL_H
