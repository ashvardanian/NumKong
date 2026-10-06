/**
 *  @file include/numkong/each/rocm.cuh
 *  @author Ash Vardanian
 *  @date October 3, 2026
 *  @brief ROCm host side of element-wise sums, RMSNorm and SwiGLU: the launches, the export
 *      generators and the @c rocm exports, over the kernels of `each/simt.cuh`.
 *
 *  @sa include/numkong/each/simt.cuh
 *  @sa include/numkong/each/cuda.cuh
 */
#ifndef NUMKONG_EACH_ROCM_CUH
#define NUMKONG_EACH_ROCM_CUH

#include "numkong/rocm.cuh"
#include "numkong/each/simt.cuh"

#if NUMKONG_ARCH_ROCM_

#if defined(__cplusplus)
extern "C" {
#endif

/** Launches @p blocks_wanted blocks of @p threads running @p kernel, but at most four times as many
 *  as stay resident across the stream's device, passing the argument struct at @p arguments. */
NUMKONG_INLINE nk_status_t nk_each_launch_rocm_(void const *kernel, unsigned threads, nk_size_t blocks_wanted,
                                                void *arguments, void *stream) {
    int caller = 0;
    nk_status_t const entered = nk_device_enter_rocm_(stream, &caller);
    if (entered != nk_success_k) return entered;
    int device = 0, multiprocessors = 0, per_multiprocessor = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, (int)threads, 0) != hipSuccess) {
        nk_device_leave_rocm_(caller);
        return nk_device_code_mismatch_k;
    }
    nk_size_t const oversubscribed = 4 * (nk_size_t)multiprocessors * (nk_size_t)per_multiprocessor;
    nk_size_t const blocks = oversubscribed < blocks_wanted ? oversubscribed : blocks_wanted;
    void *launch_arguments[1];
    launch_arguments[0] = arguments;
    dim3 grid, block;
    grid.x = (unsigned)blocks, grid.y = 1, grid.z = 1;
    block.x = threads, block.y = 1, block.z = 1;
    hipError_t const status = blocks ? hipLaunchKernel(kernel, grid, block, launch_arguments, 0, (hipStream_t)stream)
                                     : hipErrorInvalidConfiguration;
    nk_device_leave_rocm_(caller);
    return status == hipSuccess ? nk_success_k : nk_device_code_mismatch_k;
}

NUMKONG_INLINE nk_status_t nk_each_swiglu_launch_rocm_(void const *kernel, nk_size_t value_bytes, void const *gate,
                                                       void const *up, void *y, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride,
                                                       nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {
    if ((((nk_size_t)gate) | gate_stride | ((nk_size_t)up) | (up ? up_stride : 0) | ((nk_size_t)y) | y_stride) &
        (value_bytes - 1))
        return nk_misaligned_k;
    if (rows == 0 || columns == 0) return nk_success_k;
    nk_each_swiglu_arguments_t arguments;
    arguments.gate = (unsigned char const *)gate, arguments.up = (unsigned char const *)up;
    arguments.y = (unsigned char *)y, arguments.rows = rows, arguments.columns = columns;
    arguments.gate_stride = gate_stride, arguments.up_stride = up_stride, arguments.y_stride = y_stride;
    arguments.gate_scale = gate_scale, arguments.output_scale = output_scale;
    nk_size_t const block_teams = nk_each_threads_simt_k / nk_each_team_lanes_simt_k;
    return nk_each_launch_rocm_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(rows, block_teams), &arguments,
                                stream);
}

/** Validates the contract, @p value_bytes alignment for the vectors and @p scale_bytes for @p alpha
 *  and @p beta, and launches @p kernel over @p n elements. */
NUMKONG_INLINE nk_status_t nk_each_elementwise_launch_rocm_(void const *kernel, nk_size_t value_bytes,
                                                            nk_size_t scale_bytes, void const *a, void const *b,
                                                            void const *c, void const *alpha, void const *beta,
                                                            nk_size_t n, void *result, void *stream) {
    nk_size_t const addresses = ((nk_size_t)a) | ((nk_size_t)b) | ((nk_size_t)c) | ((nk_size_t)result);
    if ((addresses & (value_bytes - 1)) || (((nk_size_t)alpha) | ((nk_size_t)beta)) & (scale_bytes - 1))
        return nk_misaligned_k;
    if (n == 0) return nk_success_k;
    nk_each_elementwise_arguments_t arguments;
    arguments.a = (unsigned char const *)a, arguments.b = (unsigned char const *)b;
    arguments.c = (unsigned char const *)c, arguments.alpha = alpha, arguments.beta = beta;
    arguments.result = (unsigned char *)result, arguments.count = n;
    nk_size_t const block_values = nk_each_threads_simt_k * (addresses & 15 ? 1 : 16 / value_bytes);
    return nk_each_launch_rocm_(kernel, nk_each_threads_simt_k, nk_size_divide_round_up_(n, block_values), &arguments,
                                stream);
}

/** Validates the contract and launches one block per normalized vector, of enough whole wavefronts
 *  that each thread keeps at most @c nk_each_kept_pieces_simt_k 16-byte pieces of it. */
NUMKONG_INLINE nk_status_t nk_each_rmsnorm_launch_rocm_(void const *kernel, nk_size_t input_bytes,
                                                        nk_size_t output_bytes, void const *x, nk_f32_t const *gamma,
                                                        void *y, nk_size_t rows, nk_size_t groups, nk_size_t columns,
                                                        nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon,
                                                        void *stream) {
    nk_size_t const narrower_bytes = input_bytes < output_bytes ? input_bytes : output_bytes;
    nk_size_t const chunk_values = 16 / narrower_bytes, chunk_pieces = chunk_values * input_bytes / 16;
    if ((((nk_size_t)x) | x_stride) & (input_bytes - 1) || (((nk_size_t)y) | y_stride) & (output_bytes - 1) ||
        ((nk_size_t)gamma & 3))
        return nk_misaligned_k;
    nk_each_rmsnorm_arguments_t arguments;
    arguments.x = (unsigned char const *)x, arguments.gamma = gamma, arguments.y = (unsigned char *)y;
    arguments.groups = groups, arguments.columns = columns, arguments.vectors = rows * groups;
    arguments.x_stride = x_stride, arguments.y_stride = y_stride, arguments.epsilon = epsilon;
    if (arguments.vectors == 0 || columns == 0) return nk_success_k;
    // CDNA wavefronts hold 64 lanes, two teams each
    nk_size_t const chunks = nk_size_divide_round_up_(columns, chunk_values);
    nk_size_t const kept_chunks = nk_each_kept_pieces_simt_k / chunk_pieces;
    nk_size_t threads = nk_size_round_up_to_multiple_(nk_size_divide_round_up_(chunks, kept_chunks), 64);
    if (threads > nk_each_threads_simt_k) threads = nk_each_threads_simt_k;
    return nk_each_launch_rocm_(kernel, (unsigned)threads, arguments.vectors, &arguments, stream);
}

/** Generates the SwiGLU entry point of @p input_type over the kernel of
 *  @c nk_define_each_swiglu_kernel_simt_. */
#define nk_define_each_swiglu_rocm_(input_type, isa_suffix, load_and_convert, convert_and_store)                   \
    nk_define_each_swiglu_kernel_simt_(input_type, isa_suffix, load_and_convert, convert_and_store)                \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_swiglu_##input_type##_##isa_suffix(nk_##input_type##_t const *gate, nk_##input_type##_t const *up, \
                                                   nk_##input_type##_t *y, nk_size_t rows, nk_size_t columns,      \
                                                   nk_size_t gate_stride, nk_size_t up_stride, nk_size_t y_stride, \
                                                   nk_f32_t gate_scale, nk_f32_t output_scale, void *stream) {     \
        return nk_each_swiglu_launch_rocm_((void const *)&nk_each_swiglu_##input_type##_##isa_suffix##_kernel_,    \
                                           sizeof(nk_##input_type##_t), gate, up, y, rows, columns, gate_stride,   \
                                           up_stride, y_stride, gate_scale, output_scale, stream);                 \
    }

/** Generates the element-wise sum entry point of @p input_type over the kernel of
 *  @c nk_define_each_sum_kernel_simt_. */
#define nk_define_each_sum_rocm_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store)    \
    nk_define_each_sum_kernel_simt_(input_type, accumulator_type, isa_suffix, load_and_convert, convert_and_store) \
        NUMKONG_API nk_status_t                                                                                    \
        nk_each_sum_##input_type##_##isa_suffix(nk_##input_type##_t const *a, nk_##input_type##_t const *b,        \
                                                nk_size_t n, nk_##input_type##_t *result, void *stream) {          \
        return nk_each_elementwise_launch_rocm_((void const *)&nk_each_sum_##input_type##_##isa_suffix##_kernel_,  \
                                                sizeof(nk_##input_type##_t), 1, a, b, NUMKONG_NULL, NUMKONG_NULL,  \
                                                NUMKONG_NULL, n, result, stream);                                  \
    }

/** Generates the entry point of @p verb, an RMSNorm of @p input_type into @p output_type, over the
 *  kernel of @c nk_define_each_rmsnorm_kernel_simt_. */
#define nk_define_each_rmsnorm_rocm_(verb, input_type, output_type, isa_suffix, load_and_convert, convert_and_store)   \
    nk_define_each_rmsnorm_kernel_simt_(verb, input_type, output_type, isa_suffix, load_and_convert,                   \
                                        convert_and_store) NUMKONG_API nk_status_t                                     \
    nk_each_##verb##_##output_type##_##isa_suffix(                                                                     \
        nk_##input_type##_t const *x, nk_f32_t const *gamma, nk_##output_type##_t *y, nk_size_t rows,                  \
        nk_size_t groups, nk_size_t columns, nk_size_t x_stride, nk_size_t y_stride, nk_f32_t epsilon, void *stream) { \
        return nk_each_rmsnorm_launch_rocm_((void const *)&nk_each_##verb##_##output_type##_##isa_suffix##_kernel_,    \
                                            sizeof(nk_##input_type##_t), sizeof(nk_##output_type##_t), x, gamma, y,    \
                                            rows, groups, columns, x_stride, y_stride, epsilon, stream);               \
    }

#if NUMKONG_TARGET_ROCM
nk_define_each_sum_rocm_(f32, f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_sum_rocm_(f16, f32, rocm, nk_f16_to_f32_simt_, nk_f32_to_f16_simt_)
nk_define_each_sum_rocm_(bf16, f32, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_rocm_(rmsnorm, f32, f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_rmsnorm_rocm_(rmsnorm, bf16, bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_rocm_(rmsnorm, e4m3, e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, bf16, rocm, nk_assign_from_to_, nk_f32_to_bf16_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, f16, rocm, nk_assign_from_to_, nk_f32_to_f16_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, e4m3, rocm, nk_assign_from_to_, nk_f32_to_e4m3_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, e5m2, rocm, nk_assign_from_to_, nk_f32_to_e5m2_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, e2m3, rocm, nk_assign_from_to_, nk_f32_to_e2m3_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f32, e3m2, rocm, nk_assign_from_to_, nk_f32_to_e3m2_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, f64, f32, rocm, nk_f64_to_f32_simt_, nk_assign_from_to_)
nk_define_each_rmsnorm_rocm_(rmscast, i32, i8, rocm, nk_assign_from_to_, nk_f32_to_i8_simt_)
nk_define_each_rmsnorm_rocm_(rmscast, u32, u8, rocm, nk_assign_from_to_, nk_f32_to_u8_simt_)
nk_define_each_swiglu_rocm_(f32, rocm, nk_assign_from_to_, nk_assign_from_to_)
nk_define_each_swiglu_rocm_(bf16, rocm, nk_bf16_to_f32_simt_, nk_f32_to_bf16_simt_)
nk_define_each_swiglu_rocm_(e4m3, rocm, nk_e4m3_to_f32_simt_, nk_f32_to_e4m3_simt_)
#endif // NUMKONG_TARGET_ROCM

#undef nk_define_each_sum_rocm_
#undef nk_define_each_rmsnorm_rocm_
#undef nk_define_each_swiglu_rocm_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_EACH_ROCM_CUH
